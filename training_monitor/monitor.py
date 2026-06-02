"""
Core monitoring engine for training health
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class HealthReport:
    """Training health report"""
    timestamp: datetime
    epoch: int
    status: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    severity_level: int = 0


class TrainingHealthMonitor:
    """
    Main monitoring class for tracking neural network training health.
    """
    
    def __init__(
        self,
        model: Any,
        framework: Optional[str] = None,
        verbose: bool = True,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.framework = framework or self._detect_framework(model)
        self.verbose = verbose
        
        # Default thresholds
        self.thresholds = {
            'overfitting_ratio': 1.2,
            'underfitting_threshold': 0.5,
            'gradient_threshold': 10.0,
        }
        if thresholds:
            self.thresholds.update(thresholds)
        
        self.history = pd.DataFrame()
        self.reports: List[HealthReport] = []
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect framework"""
        model_type = str(type(model))
        if 'torch' in model_type:
            return 'pytorch'
        elif 'tensorflow' in model_type or 'keras' in model_type:
            return 'tensorflow'
        else:
            return 'unknown'
    
    def check_health(
        self,
        train_loss: float,
        val_loss: float,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        epoch: int = 0,
        gradients: Optional[np.ndarray] = None
    ) -> HealthReport:
        """Check training health at current step"""
        
        report = HealthReport(
            timestamp=datetime.now(),
            epoch=epoch
        )
        
        issues = []
        
        # Check overfitting
        if train_loss > 0:
            ratio = val_loss / train_loss
            if ratio > self.thresholds['overfitting_ratio']:
                issues.append('Overfitting detected')
                report.severity_level = max(report.severity_level, 1)
        
        # Check underfitting
        if train_loss > self.thresholds['underfitting_threshold'] and val_loss > self.thresholds['underfitting_threshold']:
            issues.append('Underfitting detected')
            report.severity_level = max(report.severity_level, 1)
        
        # Check gradients
        if gradients is not None:
            grad_issue = self._check_gradients(gradients)
            if grad_issue:
                issues.append(grad_issue)
                report.severity_level = max(report.severity_level, 2)
        
        report.issues = issues
        
        # Generate recommendations
        report.recommendations = self._get_recommendations(issues)
        
        # Set status
        if report.severity_level == 0:
            report.status = 'healthy'
        elif report.severity_level == 1:
            report.status = 'warning'
        else:
            report.status = 'critical'
        
        self.reports.append(report)
        
        # Print if verbose
        if self.verbose and issues:
            self._print_report(report)
        
        return report
    
    def _check_gradients(self, gradients: np.ndarray) -> Optional[str]:
        """Check for gradient problems"""
        if gradients is None or len(gradients) == 0:
            return None
        
        gradients = np.asarray(gradients).flatten()
        non_zero_grads = gradients[gradients != 0]
        
        if len(non_zero_grads) == 0:
            return "Vanishing gradients - all gradients are zero"
        
        grad_mean = np.mean(np.abs(non_zero_grads))
        grad_max = np.max(np.abs(non_zero_grads))
        
        if grad_mean < 1e-7:
            return "Vanishing gradients - gradients too small"
        
        if grad_max > self.thresholds['gradient_threshold']:
            return "Exploding gradients - gradients too large"
        
        if np.any(np.isnan(non_zero_grads)) or np.any(np.isinf(non_zero_grads)):
            return "Invalid gradients - NaN or Inf detected"
        
        return None
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations based on issues"""
        recommendations = {
            'Overfitting detected': [
                'Add Dropout layers (rate: 0.3-0.5)',
                'Use Early Stopping',
                'Increase training data',
                'Apply regularization (L1/L2)'
            ],
            'Underfitting detected': [
                'Increase model complexity',
                'Train for more epochs',
                'Reduce regularization',
                'Use a more powerful architecture'
            ],
            'Vanishing gradients - all gradients are zero': [
                'Add BatchNormalization layers',
                'Use ReLU instead of sigmoid',
                'Check weight initialization'
            ],
            'Exploding gradients - gradients too large': [
                'Apply gradient clipping',
                'Reduce learning rate',
                'Use BatchNormalization'
            ]
        }
        
        recs = []
        for issue in issues:
            if issue in recommendations:
                recs.extend(recommendations[issue])
        
        return recs[:10]
    
    def _print_report(self, report: HealthReport):
        """Print report"""
        print(f"\n{'='*60}")
        print(f"Epoch {report.epoch} - Status: {report.status.upper()}")
        print(f"{'='*60}")
        if report.issues:
            print("⚠️  Issues:")
            for issue in report.issues:
                print(f"  • {issue}")
        if report.recommendations:
            print("\n💡 Recommendations:")
            for rec in report.recommendations[:5]:
                print(f"  • {rec}")
        print(f"{'='*60}\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.reports:
            return {}
        
        total = len(self.reports)
        healthy = sum(1 for r in self.reports if r.status == 'healthy')
        warnings = sum(1 for r in self.reports if r.status == 'warning')
        critical = sum(1 for r in self.reports if r.status == 'critical')
        
        return {
            'total_epochs': total,
            'healthy': healthy,
            'warnings': warnings,
            'critical': critical,
            'health_percentage': (healthy / total * 100) if total > 0 else 0
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        return {
            'summary': self.get_summary(),
            'history': self.history.to_dict(),
            'latest_report': self.reports[-1].__dict__ if self.reports else None
        }
