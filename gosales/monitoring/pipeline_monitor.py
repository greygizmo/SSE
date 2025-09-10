"""
Monitoring and observability for the GoSales data pipeline.
Provides comprehensive tracking of pipeline health and performance.
"""
import time
from datetime import datetime
from typing import Dict, List, Any, Union
import pandas as pd
import polars as pl
import logging

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Comprehensive pipeline monitoring and observability."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self.stage_timings: Dict[str, float] = {}

    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage."""
        self.stage_timings[stage_name] = time.time()
        logger.info(f"Started stage: {stage_name}")

    def end_stage(self, stage_name: str, success: bool = True, details: str = None):
        """Mark the end of a pipeline stage."""
        if stage_name in self.stage_timings:
            duration = time.time() - self.stage_timings[stage_name]
            self.metrics[f"{stage_name}_duration"] = duration

            status = "completed" if success else "failed"
            logger.info(f"Stage {stage_name} {status} in {duration:.2f}s")

            if details:
                logger.info(f"Stage {stage_name} details: {details}")

        # Track stage completion
        self.metrics[f"{stage_name}_success"] = success

    def track_data_flow(self, stage: str, record_count: int, schema_info: Dict[str, Any]):
        """Track data flow metrics for a pipeline stage."""
        self.metrics[f"{stage}_records"] = record_count
        self.metrics[f"{stage}_schema"] = schema_info
        self.metrics[f"{stage}_timestamp"] = datetime.now().isoformat()

        logger.info(f"Stage {stage}: {record_count:,} records processed")

    def validate_type_consistency(self, df_dict: Dict[str, Union[pd.DataFrame, pl.DataFrame]]):
        """
        Validate type consistency across DataFrames.
        Focuses on customer_id type consistency.
        """
        customer_id_types = {}

        for name, df in df_dict.items():
            if df is None:
                continue

            try:
                if 'customer_id' in df.columns:
                    if hasattr(df, 'with_columns'):  # polars
                        customer_id_types[name] = str(df['customer_id'].dtype)
                    else:  # pandas
                        customer_id_types[name] = str(df['customer_id'].dtype)
            except Exception as e:
                logger.warning(f"Could not get customer_id type for {name}: {e}")

        # Check for inconsistencies
        if customer_id_types:
            unique_types = set(customer_id_types.values())
            if len(unique_types) > 1:
                self.alerts.append({
                    'type': 'TYPE_INCONSISTENCY',
                    'severity': 'HIGH',
                    'details': customer_id_types,
                    'message': f"Found {len(unique_types)} different customer_id types: {unique_types}",
                    'timestamp': datetime.now().isoformat()
                })
                logger.error(f"Customer ID type inconsistency detected: {customer_id_types}")
            else:
                logger.info(f"Customer ID type consistency verified: {list(unique_types)[0]}")

    def add_alert(self, alert_type: str, severity: str, message: str, details: Any = None):
        """Add an alert to the monitoring system."""
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)

        # Log based on severity
        if severity == 'HIGH':
            logger.error(f"ALERT ({alert_type}): {message}")
        elif severity == 'MEDIUM':
            logger.warning(f"ALERT ({alert_type}): {message}")
        else:
            logger.info(f"ALERT ({alert_type}): {message}")

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Generate a summary of pipeline execution."""
        total_duration = time.time() - self.start_time

        # Count alerts by severity
        alert_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for alert in self.alerts:
            severity = alert.get('severity', 'LOW')
            alert_counts[severity] += 1

        # Check for critical issues
        has_critical_issues = alert_counts['HIGH'] > 0

        summary = {
            'total_duration': total_duration,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'alert_counts': alert_counts,
            'has_critical_issues': has_critical_issues,
            'metrics': self.metrics,
            'alerts': self.alerts
        }

        return summary

    def log_summary(self):
        """Log a comprehensive pipeline summary."""
        summary = self.get_pipeline_summary()

        logger.info("=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Duration: {summary['total_duration']:.2f}s")
        logger.info(f"Alerts: {summary['alert_counts']}")

        if summary['has_critical_issues']:
            logger.error("CRITICAL ISSUES DETECTED - Pipeline may have failed")
        else:
            logger.info("Pipeline completed successfully")

        logger.info("=" * 50)

    def check_join_success_rate(self, total_joins: int, failed_joins: int):
        """Track join success rate."""
        if total_joins > 0:
            success_rate = ((total_joins - failed_joins) / total_joins) * 100
            self.metrics['join_success_rate'] = success_rate

            if success_rate < 95:
                self.add_alert(
                    'JOIN_FAILURE_RATE',
                    'HIGH',
                    f"Join success rate is only {success_rate:.1f}%",
                    {'total_joins': total_joins, 'failed_joins': failed_joins}
                )

    def monitor_memory_usage(self):
        """Monitor memory usage of key DataFrames."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        self.metrics['memory_usage_mb'] = memory_mb

        if memory_mb > 2000:  # 2GB threshold
            self.add_alert(
                'HIGH_MEMORY_USAGE',
                'MEDIUM',
                f"High memory usage detected: {memory_mb:.1f}MB",
                {'memory_mb': memory_mb}
            )
