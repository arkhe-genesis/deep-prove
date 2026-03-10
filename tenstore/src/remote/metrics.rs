use std::{borrow::Cow, collections::HashMap, future::Future, io};

use tokio::task;
use tracing::error;
use vise::{Counter, Global, LabeledFamily, Metrics};

#[vise::register]
pub(crate) static TASK_METRICS: Global<TaskMetrics> = Global::new();

#[derive(Metrics)]
pub(crate) struct TaskMetrics {
    #[metrics(labels = ["task_name"])]
    instrumented_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    dropped_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_first_poll_delay_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    first_poll_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_idle_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_idled_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_scheduled_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_scheduled_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_poll_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_poll_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_fast_poll_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_fast_poll_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_slow_poll_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_slow_poll_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_short_delay_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_short_delay_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_long_delay_duration_ms: LabeledFamily<Cow<'static, str>, Counter<u64>>,
    #[metrics(labels = ["task_name"])]
    total_long_delay_count: LabeledFamily<Cow<'static, str>, Counter<u64>>,
}

/// Struct to spawn and monitor tasks.
#[derive(Debug)]
pub struct TaskMonitor<R> {
    /// Monitors all spawned tasks
    join_set: task::JoinSet<R>,

    /// Associate a name with each id.
    ///
    /// Note: ids are not unique and can be reused. The vector contains the names in creation order,
    /// and are popped to handle id reuse.
    id_to_name: HashMap<task::Id, Vec<Cow<'static, str>>>,
}

impl<R> TaskMonitor<R> {
    pub fn new() -> Self {
        Self {
            join_set: Default::default(),
            id_to_name: Default::default(),
        }
    }
}

impl<R: 'static + Send> TaskMonitor<R> {
    /// Spawn and monitors `future`.
    ///
    /// This utility will:
    ///
    /// - Spawns the `future` inside a [task::JoinSet]. The lifecycle of the future is managed as follows:
    ///     - [TaskMonitor::join_next] returns the task [Result], if it finish execution with normal control flow.
    ///     - Otherwise, `join_next` returns an [Err] in case the task panics or is aborted.
    ///     - Once dropped, the [TaskMonitor] aborts all unfished tasks it owns.
    /// - Create a [tokio_metrics::TaskMonitor] to collect metrics and export it with
    ///   [metrics::Counter].
    ///     - Metrics are formatted using `component` and tagged using `task_name`.
    /// - The `future` task is annotated with the builder from [JoinSet::build_task]. The name metadata
    ///   is associated with a span for the task lifetime.
    ///     - NOTE: tokio uses a TRACE level span, the metadata is only available when
    ///       `tokio::task=TRACE` is enabled.
    pub fn spawn<F>(
        &mut self,
        task_name: Cow<'static, str>,
        future: F,
    ) -> io::Result<task::AbortHandle>
    where
        F: Future<Output = R> + Send + 'static,
    {
        let metrics_monitor = tokio_metrics::TaskMonitor::new();
        let instrumented = metrics_monitor.instrument(future);

        let abort_handle = self
            .join_set
            .build_task()
            .name(&task_name)
            .spawn(instrumented)?;

        self.id_to_name
            .entry(abort_handle.id())
            .or_default()
            .push(task_name.clone());

        let is_alive_handle = abort_handle.clone();

        // ignore errors with the monitoring task
        tokio::spawn(async move {
            // update task metrics every 500ms
            let frequency = std::time::Duration::from_millis(500);
            for interval in metrics_monitor.intervals() {
                let tokio_metrics::TaskMetrics {
                    instrumented_count,
                    dropped_count,
                    first_poll_count,
                    total_first_poll_delay,
                    total_idled_count,
                    total_idle_duration,
                    max_idle_duration: _,
                    total_scheduled_count,
                    total_scheduled_duration,
                    total_poll_count,
                    total_poll_duration,
                    total_fast_poll_count,
                    total_fast_poll_duration,
                    total_slow_poll_count,
                    total_slow_poll_duration,
                    total_short_delay_count,
                    total_long_delay_count,
                    total_short_delay_duration,
                    total_long_delay_duration,
                    ..
                } = interval;
                // stop the monitor when monitored task finished
                if is_alive_handle.is_finished() {
                    break;
                }

                TASK_METRICS.instrumented_count[&task_name].inc_by(instrumented_count);
                TASK_METRICS.dropped_count[&task_name].inc_by(dropped_count);

                TASK_METRICS.total_first_poll_delay_ms[&task_name]
                    .inc_by(total_first_poll_delay.as_millis() as u64);
                TASK_METRICS.first_poll_count[&task_name].inc_by(first_poll_count);

                TASK_METRICS.total_idle_duration_ms[&task_name]
                    .inc_by(total_idle_duration.as_millis() as u64);
                TASK_METRICS.total_idled_count[&task_name].inc_by(total_idled_count);

                TASK_METRICS.total_scheduled_duration_ms[&task_name]
                    .inc_by(total_scheduled_duration.as_millis() as u64);
                TASK_METRICS.total_scheduled_count[&task_name].inc_by(total_scheduled_count);

                TASK_METRICS.total_poll_duration_ms[&task_name]
                    .inc_by(total_poll_duration.as_millis() as u64);
                TASK_METRICS.total_poll_count[&task_name].inc_by(total_poll_count);

                TASK_METRICS.total_fast_poll_duration_ms[&task_name]
                    .inc_by(total_fast_poll_duration.as_millis() as u64);
                TASK_METRICS.total_fast_poll_count[&task_name].inc_by(total_fast_poll_count);

                TASK_METRICS.total_slow_poll_duration_ms[&task_name]
                    .inc_by(total_slow_poll_duration.as_millis() as u64);
                TASK_METRICS.total_slow_poll_count[&task_name].inc_by(total_slow_poll_count);

                TASK_METRICS.total_short_delay_duration_ms[&task_name]
                    .inc_by(total_short_delay_duration.as_millis() as u64);
                TASK_METRICS.total_short_delay_count[&task_name].inc_by(total_short_delay_count);

                TASK_METRICS.total_long_delay_duration_ms[&task_name]
                    .inc_by(total_long_delay_duration.as_millis() as u64);
                TASK_METRICS.total_long_delay_count[&task_name].inc_by(total_long_delay_count);

                tokio::time::sleep(frequency).await;
            }
        });

        Ok(abort_handle)
    }

    /// Waits until one of the tasks in the set completes and returns its output and name.
    ///
    /// Returns `None` if the set is empty.
    pub async fn join_next(&mut self) -> Option<(Result<R, task::JoinError>, Cow<'static, str>)> {
        match self.join_set.join_next_with_id().await? {
            Ok((id, res)) => {
                let name = self
                    .id_to_name
                    .get_mut(&id)
                    .expect("id must have an associated name")
                    .remove(0);

                Some((Ok(res), name))
            }
            Err(err) => {
                let name = self
                    .id_to_name
                    .get_mut(&err.id())
                    .expect("id must have an associated name")
                    .remove(0);

                Some((Err(err), name))
            }
        }
    }

    /// Ensure that all the tasks in this monitor have completed.
    pub async fn join_all(&mut self) {
        while let Some(res) = self.join_set.join_next().await {
            match res {
                Ok(_) => {}
                Err(err) => error!("while joining task monitor: {err:?}"),
            }
        }
    }
}

impl<R> Default for TaskMonitor<R> {
    fn default() -> Self {
        Self::new()
    }
}
