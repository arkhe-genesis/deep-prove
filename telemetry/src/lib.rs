#[cfg(not(feature = "otel"))]
pub use disabled::*;

#[cfg(not(feature = "otel"))]
mod disabled {
    use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt, layer::SubscriberExt, registry};
    use ureq::RequestBuilder;

    #[derive(Clone, Copy)]
    pub struct OtelGuard;

    pub fn setup_logging(_service_name: &str, json: bool) -> Option<OtelGuard> {
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        if json {
            tracing::subscriber::set_global_default(
                registry().with(filter).with(
                    fmt::layer()
                        .json()
                        .with_level(true)
                        .with_file(true)
                        .with_line_number(true)
                        .with_target(true),
                ),
            )
            .expect("Setting up logging failed");
        } else {
            tracing::subscriber::set_global_default(
                registry().with(filter).with(
                    fmt::layer()
                        .pretty()
                        .compact()
                        .with_level(true)
                        .with_file(true)
                        .with_line_number(true)
                        .with_target(true),
                ),
            )
            .expect("Setting up logging failed");
        };

        None
    }

    pub fn inject_trace_headers<T>(req: RequestBuilder<T>) -> RequestBuilder<T> {
        req
    }

    pub fn set_parent_from_headers(_span: &tracing::Span, _headers: &[(String, String)]) {}
}

#[cfg(feature = "otel")]
pub use enabled::*;

#[cfg(feature = "otel")]
mod enabled {
    use std::{env, time::Duration};

    use opentelemetry::{
        Context, KeyValue,
        global::{get_text_map_propagator, set_text_map_propagator, set_tracer_provider},
        propagation::{Extractor, Injector},
        trace::{TraceContextExt, TracerProvider as _},
    };
    use opentelemetry_otlp::{self, Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
    use opentelemetry_sdk::{
        Resource, runtime,
        propagation::TraceContextPropagator,
        trace::{
            SdkTracer, SdkTracerProvider, span_processor_with_async_runtime::BatchSpanProcessor,
        },
    };
    use opentelemetry_semantic_conventions::resource;
    use reqwest::Client;
    use tracing::{Span, field};
    use tracing_opentelemetry::{OpenTelemetryLayer, OpenTelemetrySpanExt, layer};
    use tracing_subscriber::{
        EnvFilter, Layer, Registry, filter::LevelFilter, fmt, layer::SubscriberExt, registry,
    };
    use ureq::RequestBuilder;

    type OtelLayer = OpenTelemetryLayer<Registry, SdkTracer>;

    /// Guard that keeps the tracer provider alive and shuts it down on drop.
    pub struct OtelGuard(SdkTracerProvider);

    impl Drop for OtelGuard {
        fn drop(&mut self) {
            let _ = self.0.shutdown();
        }
    }

    /// Initialize OTEL layer if `OTEL_EXPORTER_OTLP_ENDPOINT` is set otherwise returns `None`.
    fn init_otel_layer(service_name: &str) -> Option<(OtelLayer, OtelGuard)> {
        let endpoint = match env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            Ok(s) if !s.is_empty() => reqwest::Url::parse(&s)
                .and_then(|url| url.join("/v1/traces"))
                .ok()?
                .to_string(),
            _ => {
                eprintln!("OTEL tracing disabled (no OTEL_EXPORTER_OTLP_ENDPOINT set)");
                return None;
            }
        };

        let service_name = env::var("OTEL_SERVICE_NAME")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| service_name.to_string());
        let host_name = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
        let resource = Resource::builder_empty()
            .with_attributes([
                KeyValue::new(resource::SERVICE_NAME, service_name),
                KeyValue::new(resource::SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                KeyValue::new("host.name", host_name),
            ])
            .build();

        let exporter = match SpanExporter::builder()
            .with_http()
            .with_http_client(Client::new())
            .with_endpoint(endpoint)
            .with_timeout(Duration::from_secs(10))
            .with_protocol(Protocol::HttpJson)
            .build()
        {
            Ok(exp) => exp,
            Err(err) => {
                eprintln!("Failed to build OTLP exporter; telemetry disabled: {err}");
                return None;
            }
        };

        let batch = BatchSpanProcessor::builder(exporter, runtime::Tokio).build();

        let provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .with_span_processor(batch)
            .build();

        set_tracer_provider(provider.clone());
        set_text_map_propagator(TraceContextPropagator::new());
        let tracer = provider.tracer("telemetry");
        let otel_layer = layer().with_tracer(tracer);
        let guard = OtelGuard(provider);

        eprintln!("OTEL tracing enabled; exporting to OTLP");
        Some((otel_layer, guard))
    }

    /// Initialize logging + optional OTLP tracing/export.
    pub fn setup_logging(service_name: &str, json: bool) -> Option<OtelGuard> {
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        if let Some((otel_layer, guard)) = init_otel_layer(service_name) {
            if json {
                tracing::subscriber::set_global_default(
                    registry()
                        .with(otel_layer.with_filter(LevelFilter::TRACE))
                        .with(filter.clone())
                        .with(
                            fmt::layer()
                                .json()
                                .with_level(true)
                                .with_file(true)
                                .with_line_number(true)
                                .with_target(true)
                                .event_format(OtelLogFormatter),
                        ),
                )
                .expect("Setting up logging failed");
                return Some(guard);
            } else {
                tracing::subscriber::set_global_default(
                    registry()
                        .with(otel_layer.with_filter(LevelFilter::TRACE))
                        .with(filter.clone())
                        .with(
                            fmt::layer()
                                .pretty()
                                .compact()
                                .with_level(true)
                                .with_file(true)
                                .with_line_number(true)
                                .with_target(true)
                                .event_format(OtelLogFormatter),
                        ),
                )
                .expect("Setting up logging failed");
                return Some(guard);
            }
        }
        None
    }

    /// Inject current context into a carrier (HTTP headers, etc).
    fn inject_context(ctx: &Context, carrier: &mut impl Injector) {
        get_text_map_propagator(|prop| prop.inject_context(ctx, carrier));
    }

    /// Extract context from a carrier (HTTP headers, etc).
    fn extract_context(carrier: &impl Extractor) -> Option<Context> {
        let ctx = get_text_map_propagator(|prop| prop.extract(carrier));
        if ctx.span().span_context().is_valid() {
            Some(ctx)
        } else {
            None
        }
    }

    /// Returns headers carrying the current tracing context (traceparent/baggage).
    /// Empty when no valid span is present.
    fn current_trace_headers() -> Vec<(String, String)> {
        struct Collector<'a>(&'a mut Vec<(String, String)>);
        impl<'a> Injector for Collector<'a> {
            fn set(&mut self, key: &str, value: String) {
                if key.eq_ignore_ascii_case("traceparent") {
                    self.0.push((key.to_string(), value));
                }
            }
        }

        let ctx = tracing::Span::current().context();
        let mut headers = Vec::new();
        inject_context(&ctx, &mut Collector(&mut headers));
        headers
    }

    /// Inject the current trace headers into a ureq request builder.
    pub fn inject_trace_headers<T>(req: RequestBuilder<T>) -> RequestBuilder<T> {
        let mut req = req;
        for (k, v) in current_trace_headers() {
            req = req.header(&k, &v);
        }
        req
    }

    /// Sets the OpenTelemetry parent for a span from incoming headers.
    pub fn set_parent_from_headers(span: &tracing::Span, headers: &[(String, String)]) {
        struct HeaderExtractor<'a> {
            headers: &'a [(String, String)],
        }
        impl<'a> Extractor for HeaderExtractor<'a> {
            fn get(&self, key: &str) -> Option<&str> {
                self.headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(key))
                    .map(|(_, v)| v.as_str())
            }

            fn keys(&self) -> Vec<&str> {
                self.headers.iter().map(|(k, _)| k.as_str()).collect()
            }
        }

        if let Some(ctx) = extract_context(&HeaderExtractor { headers }) {
            let _ = span.set_parent(ctx);
        }
    }

    /// Formatter that prefixes log events with trace/span IDs when available.
    #[derive(Debug, Default, Clone, Copy)]
    struct OtelLogFormatter;
    impl<S, N> fmt::FormatEvent<S, N> for OtelLogFormatter
    where
        S: tracing::Subscriber + for<'a> registry::LookupSpan<'a>,
        N: for<'writer> fmt::FormatFields<'writer> + 'static,
    {
        fn format_event(
            &self,
            _ctx: &fmt::FmtContext<'_, S, N>,
            mut writer: fmt::format::Writer<'_>,
            event: &tracing::Event<'_>,
        ) -> std::fmt::Result {
            use std::fmt::Write as _;

            let span_ctx = Span::current().context().span().span_context().clone();
            if span_ctx.is_valid() {
                write!(
                    &mut writer,
                    "[trace_id={}, span_id={}] ",
                    span_ctx.trace_id(),
                    span_ctx.span_id()
                )?;
            }

            let meta = event.metadata();
            write!(&mut writer, "{} {} ", meta.level(), meta.target())?;

            // Collect event fields into a flat `key=value` string.
            let mut fields = String::new();
            struct FieldVisitor<'a> {
                buf: &'a mut String,
                first: bool,
            }
            impl<'a> field::Visit for FieldVisitor<'a> {
                fn record_debug(&mut self, field: &field::Field, value: &dyn std::fmt::Debug) {
                    if !self.first {
                        let _ = write!(self.buf, " ");
                    }
                    self.first = false;
                    let _ = write!(self.buf, "{}={value:?}", field.name());
                }
            }
            event.record(&mut FieldVisitor {
                buf: &mut fields,
                first: true,
            });

            if !fields.is_empty() {
                write!(&mut writer, "{fields}")?;
            }
            writeln!(&mut writer)
        }
    }
}
