import numpy as np
import math
from enum import Enum
from collections import deque
from typing import Optional, Tuple, List

# Constants
HIGHEST_POSSIBLE_BREATHING_RATE = 45
LOWEST_POSSIBLE_BREATHING_RATE = 5
NUMBER_OF_ABNORMAL_BREATHS_SWITCH = 3

class ThresholdValueType(Enum):
    POSITIVE = "positive"
    INVALID = "invalid"
    NEGATIVE = "negative"

class BpmState(Enum):
    LOW = "low"
    MID_FALLING = "mid_falling"
    MID_UNKNOWN = "mid_unknown"
    MID_RISING = "mid_rising"
    HIGH = "high"
    UNKNOWN = "unknown"

class ThresholdBuffer:
    """Adaptive threshold calculation using RMS of positive/negative values"""
    
    def __init__(self, threshold_filter_size: int):
        self.threshold_filter_size = threshold_filter_size
        self.fill = 0
        self.current_position = -1
        self.is_valid = False
        self.lower_values_sum = 0.0
        self.upper_values_sum = 0.0
        self.upper_values_sum_fill = 0
        self.lower_values_sum_fill = 0
        self.upper_threshold_value = float('nan')
        self.lower_threshold_value = float('nan')
        
        self.values = [0.0] * threshold_filter_size
        self.values_type = [ThresholdValueType.INVALID] * threshold_filter_size
    
    def update_rms_threshold(self, breathing_signal_value: float):
       
        self.current_position = (self.current_position + 1) % self.threshold_filter_size
        if self.values_type[self.current_position] == ThresholdValueType.POSITIVE:
            self.upper_values_sum -= self.values[self.current_position]
            self.upper_values_sum_fill -= 1
        elif self.values_type[self.current_position] == ThresholdValueType.NEGATIVE:
            self.lower_values_sum -= self.values[self.current_position]
            self.lower_values_sum_fill -= 1
        
        if math.isnan(breathing_signal_value):
            self.values_type[self.current_position] = ThresholdValueType.INVALID
        else:
            squared_value = breathing_signal_value * breathing_signal_value
            self.values[self.current_position] = squared_value
        
            if breathing_signal_value >= 0:
                self.upper_values_sum_fill += 1
                self.values_type[self.current_position] = ThresholdValueType.POSITIVE
                self.upper_values_sum += squared_value
            else:
                self.lower_values_sum_fill += 1
                self.values_type[self.current_position] = ThresholdValueType.NEGATIVE
                self.lower_values_sum += squared_value
        
        if self.fill < self.threshold_filter_size:
            self.fill += 1
        
        if self.fill < self.threshold_filter_size:
            self.is_valid = False
            return
        
        if self.upper_values_sum_fill > 0:
            self.upper_threshold_value = math.sqrt(
                self.upper_values_sum / self.upper_values_sum_fill
            )
        else:
            self.upper_threshold_value = float('nan')
        
        # Calculate current lower threshold
        if self.lower_values_sum_fill > 0:
            # Calculate the root mean square
            self.lower_threshold_value = -math.sqrt(
                self.lower_values_sum / self.lower_values_sum_fill
            )
        else:
            self.lower_threshold_value = float('nan')
        
        self.is_valid = True

class CurrentBreath:
    """State machine for detecting individual breath cycles"""
    
    def __init__(self, lower_threshold_limit: float, upper_threshold_limit: float, 
                 sampling_frequency: float):
        self.state = BpmState.UNKNOWN
        self.breathing_rate = float('nan')
        self.min_threshold = lower_threshold_limit
        self.max_threshold = upper_threshold_limit
        self.sample_count = 0
        self.sampling_frequency = sampling_frequency
        self.is_current_breath_valid = False
        self.is_complete = False
        
        self.is_inspiration_above_x = True
        self.first_part_length = 0
        self.count_abnormal_breaths = 0
        self.completed_breath_sample_count = 0
    
    def end_breath(self):
        """Process the end of a complete breath cycle"""
        # Store the sample count for this completed breath before resetting
        sample_count_for_this_breath = self.sample_count
        
        if self.is_current_breath_valid:
            if self.first_part_length > self.sample_count - self.first_part_length:
                self.count_abnormal_breaths += 1
            else:
                # Reset count
                self.count_abnormal_breaths = 0
            
            # If we have three abnormal breaths, the breath detection is "flipped"
            if self.count_abnormal_breaths >= NUMBER_OF_ABNORMAL_BREATHS_SWITCH:
                self.count_abnormal_breaths = 0
                self.is_inspiration_above_x = not self.is_inspiration_above_x
            else:
                # Only when we didn't have 3 abnormal breaths in a row do we count this breath as valid
                new_breathing_rate = 60.0 * self.sampling_frequency / float(self.sample_count)
                
                # We want the breathing rate to lie in a realistic range
                if (new_breathing_rate >= LOWEST_POSSIBLE_BREATHING_RATE and 
                    new_breathing_rate <= HIGHEST_POSSIBLE_BREATHING_RATE):
                    self.breathing_rate = new_breathing_rate
                    self.is_complete = True
                    self.completed_breath_sample_count = sample_count_for_this_breath
        
        self.sample_count = 0
        self.first_part_length = 0
        self.is_current_breath_valid = True
    
    def update_breath(self, breathing_signal: float, upper_threshold: float, 
                     lower_threshold: float):
        """Update breath detection state machine with new signal value"""
        self.sample_count += 1
        
        if (math.isnan(upper_threshold) or math.isnan(lower_threshold) or 
            math.isnan(breathing_signal)):
            self.breathing_rate = float('nan')
            self.is_current_breath_valid = False
            return
        
        if self.state == BpmState.UNKNOWN:
            if breathing_signal < lower_threshold:
                self.state = BpmState.LOW
            elif breathing_signal > upper_threshold:
                self.state = BpmState.HIGH
            else:
                self.state = BpmState.MID_UNKNOWN
        
        if upper_threshold - lower_threshold < self.min_threshold * 2.0:
            self.state = BpmState.UNKNOWN
            self.breathing_rate = float('nan')
            self.is_current_breath_valid = False
            return
        
        if upper_threshold - lower_threshold > self.max_threshold * 2.0:
            self.state = BpmState.UNKNOWN
            self.breathing_rate = float('nan')
            self.is_current_breath_valid = False
            return
        
        if self.state == BpmState.LOW and breathing_signal > lower_threshold:
            self.state = BpmState.MID_RISING
        elif self.state == BpmState.HIGH and breathing_signal < upper_threshold:
            self.state = BpmState.MID_FALLING
        elif ((self.state == BpmState.MID_RISING or self.state == BpmState.MID_UNKNOWN) and 
              breathing_signal > upper_threshold):
            self.state = BpmState.HIGH
            
            if self.is_inspiration_above_x:
                self.end_breath()
            else:
                self.first_part_length = self.sample_count
        elif ((self.state == BpmState.MID_FALLING or self.state == BpmState.MID_UNKNOWN) and 
              breathing_signal < lower_threshold):
            self.state = BpmState.LOW

            if self.is_inspiration_above_x:
                self.first_part_length = self.sample_count
            else:
                self.end_breath()

class BreathingRateStats:
    """Statistical analysis of breathing rates with outlier removal"""
    
    BREATHING_RATES_BUFFER_SIZE = 50
    DISCARD_UPPER_BREATHING_RATES = 2
    DISCARD_LOWER_BREATHING_RATES = 2
    
    def __init__(self):
        self.fill = 0
        self.breathing_rates = [0.0] * self.BREATHING_RATES_BUFFER_SIZE
        self.is_valid = False
        self.previous_mean = 0.0
        self.current_mean = 0.0
        self.previous_variance = 0.0
        self.current_variance = 0.0
        self.max_rate = 0.0
        self.min_rate = 0.0
    
    def update_breathing_rate_stats(self, breathing_rate: float):
        """Add a new breathing rate to the buffer"""
        if self.fill < self.BREATHING_RATES_BUFFER_SIZE:
            self.breathing_rates[self.fill] = breathing_rate
            self.fill += 1
    
    def calculate_breathing_rate_stats(self):
        """Calculate statistics with outlier removal"""
        sorted_rates = sorted(self.breathing_rates[:self.fill])
        
        if self.fill <= (self.DISCARD_LOWER_BREATHING_RATES + self.DISCARD_UPPER_BREATHING_RATES):
            self.is_valid = False
            return

        start_idx = self.DISCARD_LOWER_BREATHING_RATES
        end_idx = self.fill - self.DISCARD_UPPER_BREATHING_RATES
        valid_rates = sorted_rates[start_idx:end_idx]
        
        if not valid_rates:
            self.is_valid = False
            return
        
        # Initialize with first value
        one_breathing_rate = valid_rates[0]
        self.previous_mean = self.current_mean = one_breathing_rate
        self.previous_variance = 0.0
        self.max_rate = one_breathing_rate
        self.min_rate = one_breathing_rate
        
        for i, one_breathing_rate in enumerate(valid_rates):
            n = i + 1
            self.current_mean = self.previous_mean + (one_breathing_rate - self.previous_mean) / n
            
            self.current_variance = (self.previous_variance + 
                                   (one_breathing_rate - self.previous_mean) * 
                                   (one_breathing_rate - self.current_mean))
            
            self.previous_mean = self.current_mean
            self.previous_variance = self.current_variance

            self.max_rate = max(one_breathing_rate, self.max_rate)
            self.min_rate = min(one_breathing_rate, self.min_rate)
        
        self.is_valid = True
    
    def get_mean(self) -> float:
        """Get mean breathing rate"""
        return self.current_mean if self.is_valid else float('nan')
    
    def get_variance(self) -> float:
        """Get variance of breathing rates"""
        if not self.is_valid or self.fill <= 1:
            return float('nan')
        return self.current_variance / (self.fill - 1)
    
    def get_standard_deviation(self) -> float:
        """Get standard deviation of breathing rates"""
        variance = self.get_variance()
        return math.sqrt(variance) if not math.isnan(variance) else float('nan')
    
    def get_number_of_breaths(self) -> int:
        """Get number of breaths in buffer"""
        return self.fill

class BreathDetector:
    """Main class that combines threshold calculation and breath detection"""
    
    def __init__(self, threshold_filter_size: int = 100, 
                 lower_threshold_limit: float = 0.1, 
                 upper_threshold_limit: float = 2.0,
                 sampling_frequency: float = 12.5):
        
        self.threshold_buffer = ThresholdBuffer(threshold_filter_size)
        self.current_breath = CurrentBreath(lower_threshold_limit, upper_threshold_limit, 
                                          sampling_frequency)
        self.breathing_rate_stats = BreathingRateStats()
        
        # Storage for detected breaths
        self.detected_breaths = []
        self.breath_rates = []
    
    def process_signal_value(self, breathing_signal: float) -> Optional[float]:
        """
        Process a single breathing signal value and return breathing rate if a breath is detected
        
        Args:
            breathing_signal: The processed breathing signal value
            
        Returns:
            Breathing rate in breaths per minute if a complete breath was detected, None otherwise
        """
        # Reset completion flag
        self.current_breath.is_complete = False
        
        # Update adaptive thresholds
        self.threshold_buffer.update_rms_threshold(breathing_signal)
        
        if not self.threshold_buffer.is_valid:
            return None
        
        # Update breath detection state machine
        self.current_breath.update_breath(
            breathing_signal,
            self.threshold_buffer.upper_threshold_value,
            self.threshold_buffer.lower_threshold_value
        )
        
        # Check if a complete breath was detected
        if self.current_breath.is_complete:
            breathing_rate = self.current_breath.breathing_rate
            
            # Store the detected breath
            self.detected_breaths.append({
                'rate': breathing_rate,
                'sample_count': self.current_breath.completed_breath_sample_count,
                'upper_threshold': self.threshold_buffer.upper_threshold_value,
                'lower_threshold': self.threshold_buffer.lower_threshold_value,
                'is_inspiration_above_x': self.current_breath.is_inspiration_above_x
            })
            
            # Update statistics
            self.breathing_rate_stats.update_breathing_rate_stats(breathing_rate)
            self.breath_rates.append(breathing_rate)
            
            return breathing_rate
        
        return None
    
    def process_signal_array(self, breathing_signals: List[float]) -> List[float]:
        """
        Process an array of breathing signal values
        
        Args:
            breathing_signals: List of breathing signal values
            
        Returns:
            List of detected breathing rates
        """
        detected_rates = []
        
        for signal in breathing_signals:
            rate = self.process_signal_value(signal)
            if rate is not None:
                detected_rates.append(rate)
        
        return detected_rates
    
    def get_current_statistics(self) -> dict:
        """Get current breathing rate statistics"""
        self.breathing_rate_stats.calculate_breathing_rate_stats()
        
        return {
            'mean_rate': self.breathing_rate_stats.get_mean(),
            'std_deviation': self.breathing_rate_stats.get_standard_deviation(),
            'variance': self.breathing_rate_stats.get_variance(),
            'min_rate': self.breathing_rate_stats.min_rate,
            'max_rate': self.breathing_rate_stats.max_rate,
            'num_breaths': self.breathing_rate_stats.get_number_of_breaths(),
            'is_valid': self.breathing_rate_stats.is_valid
        }
    
    def get_detected_breaths(self) -> List[dict]:
        """Get all detected breaths with their metadata"""
        return self.detected_breaths.copy()
    
    def get_current_thresholds(self) -> Tuple[float, float]:
        """Get current adaptive thresholds"""
        return (self.threshold_buffer.lower_threshold_value, 
                self.threshold_buffer.upper_threshold_value)



# Example

# detector = BreathDetector(
#     threshold_filter_size=150,      # From optimization
#     lower_threshold_limit=0.02,     # From optimization  
#     upper_threshold_limit=3.0,      # From optimization
#     sampling_frequency=12.5
# )

# # 4. Process the signal
# breathing_rates = []
# for signal_value in breathing_signal:
#     rate = detector.process_signal_value(signal_value)
#     if rate is not None:
#         breathing_rates.append(rate)

# print(f"Detected {len(breathing_rates)} breaths")
# print(f"Mean breathing rate: {np.mean(breathing_rates):.1f} BPM")