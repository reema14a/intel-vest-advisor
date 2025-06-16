import logging
import os
import time
from functools import wraps
from typing import Callable, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class Monitoring:
    """Monitoring and observability utilities using OpenTelemetry and Google Cloud services"""
    
    def __init__(self, environment: str = "dev", enable_local_monitoring: bool = True, enable_cloud_monitoring: bool = False):
        """Initialize monitoring with environment-specific settings
        
        Args:
            environment: Deployment environment (development/production)
            enable_local_monitoring: Whether to enable local monitoring (console exporters)
            enable_cloud_monitoring: Whether to enable cloud monitoring (cloud exporters)
        """
        self.environment = environment
        self.enable_local_monitoring = enable_local_monitoring
        self.enable_cloud_monitoring = enable_cloud_monitoring
        
        # Only initialize OpenTelemetry if any monitoring is enabled
        if self.enable_local_monitoring or self.enable_cloud_monitoring:
            # Initialize OpenTelemetry
            self._setup_opentelemetry()
            logger.info(f"Monitoring initialized for {environment} environment")
        else:
            logger.info(f"All monitoring disabled for {environment} environment - OpenTelemetry setup skipped")
    
    def _setup_opentelemetry(self):
        """Set up OpenTelemetry with appropriate exporters based on environment"""
        # Only proceed if any monitoring is enabled
        if not (self.enable_local_monitoring or self.enable_cloud_monitoring):
            logger.info("OpenTelemetry setup skipped - all monitoring disabled")
            return
            
        # Import OpenTelemetry modules only when needed
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        except ImportError as e:
            logger.warning(f"Failed to import OpenTelemetry modules: {e}")
            return
            
        # Create resource with service information
        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "intel-vest-advisor"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "deployment.environment": self.environment,
            "gcp.project_id": os.getenv("GCP_PROJECT_ID", "unknown"),
        })
        
        logger.info(f"Setting up OpenTelemetry with resource: {resource.attributes}")
        
        # Set up tracing
        self.tracer_provider = TracerProvider(resource=resource)
        self.tracer = self.tracer_provider.get_tracer(__name__)
        
        # Set up metrics
        metric_readers = []
        
        # Console exporters for local development (only if local monitoring is enabled)
        if self.enable_local_monitoring:
            console_span_exporter = ConsoleSpanExporter()
            console_metric_exporter = ConsoleMetricExporter()
            
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_span_exporter)
            )
            
            metric_readers.append(
                PeriodicExportingMetricReader(
                    exporter=console_metric_exporter,
                    export_interval_millis=10000  # 10 seconds
                )
            )
            
            logger.info("Console exporters configured for local monitoring")
        
        # Get OTLP endpoint from environment
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                is_cloud_trace = "cloudtrace.googleapis.com" in otlp_endpoint
                logger.info(f"Configuring OTLP exporters for endpoint: {otlp_endpoint} (secure: {is_cloud_trace})")
                
                # OTLP Trace Exporter
                otlp_trace_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    insecure=not is_cloud_trace
                )
                self.tracer_provider.add_span_processor(
                    BatchSpanProcessor(otlp_trace_exporter)
                )
                
                # OTLP Metric Exporter
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=otlp_endpoint,
                    insecure=not is_cloud_trace
                )
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        exporter=otlp_metric_exporter,
                        export_interval_millis=15000  # 15 seconds
                    )
                )
                
                logger.info("OTLP exporters configured successfully")
            except Exception as e:
                logger.error(f"Failed to set up OTLP exporters: {e}", exc_info=True)
        else:
            logger.warning("No OTLP_ENDPOINT specified in environment variables")
        
        # Set providers
        trace.set_tracer_provider(self.tracer_provider)
        
        if metric_readers:
            self.meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
            metrics.set_meter_provider(self.meter_provider)
            self.meter = self.meter_provider.get_meter(__name__)
            
            # Create custom metrics
            self._setup_custom_metrics()
            
            logger.info("Metrics provider configured with readers: %s", 
                       [reader.__class__.__name__ for reader in metric_readers])
        
        # Instrument HTTP clients and frameworks (only if monitoring is enabled)
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
            logger.debug("Requests instrumentation enabled")
        except ImportError:
            logger.debug("Requests instrumentation not available")
        except Exception as e:
            logger.warning(f"Failed to instrument requests: {e}")
        
        try:
            from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
            GrpcInstrumentorClient().instrument()
            logger.debug("gRPC instrumentation enabled")
        except ImportError:
            logger.debug("gRPC instrumentation not available")
        except Exception as e:
            logger.warning(f"Failed to instrument gRPC: {e}")
    
    def _setup_custom_metrics(self):
        """Set up custom metrics for monitoring"""
        try:
            # Create custom metrics
            self.prediction_counter = self.meter.create_counter(
                name="model_predictions_total",
                description="Total number of model predictions",
                unit="1"
            )
            
            self.prediction_latency = self.meter.create_histogram(
                name="model_prediction_latency_seconds",
                description="Model prediction latency in seconds",
                unit="s"
            )
            
            self.error_counter = self.meter.create_counter(
                name="model_errors_total",
                description="Total number of model errors",
                unit="1"
            )
            
            logger.info("Custom metrics configured")
        except Exception as e:
            logger.warning(f"Failed to set up custom metrics: {e}")
    
    def trace_function(self, name: str = None) -> Callable:
        """Decorator to trace function execution"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not (self.enable_local_monitoring or self.enable_cloud_monitoring) or not hasattr(self, 'tracer'):
                    # Monitoring disabled or tracer not initialized, just execute function
                    return func(*args, **kwargs)
                
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    # Add function arguments as span attributes
                    span.set_attribute("function.args", str(args))
                    span.set_attribute("function.kwargs", str(kwargs))
                    
                    try:
                        result = func(*args, **kwargs)
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
            return wrapper
        return decorator
    
    def log_agent_interaction(self, agent_name: str, action: str, details: dict = None):
        """Log agent interactions"""
        if not (self.enable_local_monitoring or self.enable_cloud_monitoring):
            return
            
        # Local logging
        logger.info(f"Agent {agent_name} performed {action}: {details}")
    
    def log_prediction(self, model_name: str, input_data: dict, prediction: dict):
        """Log model predictions"""
        if not (self.enable_local_monitoring or self.enable_cloud_monitoring):
            return
            
        # Local logging
        logger.info(f"Model {model_name} prediction: {prediction}")
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log errors with context"""
        if not (self.enable_local_monitoring or self.enable_cloud_monitoring):
            return
            
        # Local logging
        logger.error(f"Error {error_type}: {error_message} - Context: {context}")

# Create global monitoring instance based on environment configuration
def _get_monitoring_config():
    """Get monitoring configuration from environment variables"""
    # Import env_manager here to avoid circular imports
    from utils.env_manager import env_manager
    
    # Load environment configuration
    try:
        env_manager.load_environment()
    except Exception as e:
        logger.warning(f"Failed to load environment configuration: {e}")
    
    enable_local_monitoring = os.getenv("ENABLE_LOCAL_MONITORING", "true").lower() == "true"
    enable_cloud_monitoring = os.getenv("ENABLE_CLOUD_MONITORING", "false").lower() == "true"
    environment = os.getenv("ENVIRONMENT", "dev")
    
    logger.info(f"Monitoring config: local={enable_local_monitoring}, cloud={enable_cloud_monitoring}, env={environment}")
    
    # If both local and cloud monitoring are disabled, disable monitoring entirely
    if not enable_local_monitoring and not enable_cloud_monitoring:
        logger.info("Both local and cloud monitoring are disabled - monitoring will not be initialized")
        return None
    
    return Monitoring(
        environment=environment,
        enable_local_monitoring=enable_local_monitoring,
        enable_cloud_monitoring=enable_cloud_monitoring
    )

# Create global monitoring instance only if monitoring is enabled
monitoring = _get_monitoring_config()

# Create a no-op decorator for when monitoring is disabled
def no_op_trace_function(name: str = None) -> Callable:
    """No-op decorator when monitoring is disabled"""
    def decorator(func: Callable) -> Callable:
        return func
    return decorator

# Example usage (only if monitoring is enabled):
if monitoring:
    @monitoring.trace_function()
    def example_function():
        monitoring.log_agent_interaction("example_agent", "test_action", {"test": "data"})
        monitoring.log_prediction("example_model", {"input": "data"}, {"output": "result"})
        monitoring.log_error("test_error", "Something went wrong", {"context": "test"})
else:
    # Define a no-op function when monitoring is disabled
    @no_op_trace_function()
    def example_function():
        pass 