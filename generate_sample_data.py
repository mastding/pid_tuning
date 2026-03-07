#!/usr/bin/env python3
# Generate Sample CSV Data for Testing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(output_path='data/sample_data.csv', duration=600):
    """Generate sample control loop data"""
    
    # Time series
    dt = 1.0  # 1 second sampling
    n_points = int(duration / dt)
    timestamps = [datetime.now() + timedelta(seconds=i*dt) for i in range(n_points)]
    
    # Setpoint (SV) with step changes
    sv = np.ones(n_points) * 50
    sv[200:400] = 70  # Step change at t=200s
    sv[400:] = 60     # Another step at t=400s
    
    # Simulate process response (FOPDT-like)
    K = 1.2   # Process gain
    T = 20.0  # Time constant
    L = 5.0   # Dead time
    
    pv = np.zeros(n_points)
    mv = np.zeros(n_points)
    
    # Simple PID controller
    Kp, Ki, Kd = 0.5, 0.05, 2.0
    integral = 0
    prev_error = 0
    
    L_steps = int(L / dt)
    mv_history = [50] * (L_steps + 1)
    
    for i in range(n_points):
        # PID control
        error = sv[i] - pv[i]
        proportional = Kp * error
        integral += Ki * error * dt
        derivative = Kd * (error - prev_error) / dt if i > 0 else 0
        
        mv[i] = np.clip(proportional + integral + derivative + 50, 0, 100)
        
        # Process response with delay
        mv_history.append(mv[i])
        mv_delayed = mv_history[0]
        mv_history.pop(0)
        
        # First order dynamics
        if i > 0:
            alpha = dt / (T + dt)
            pv[i] = (1 - alpha) * pv[i-1] + K * alpha * (mv_delayed - 50) + sv[i]
        else:
            pv[i] = sv[i]
        
        # Add noise
        pv[i] += np.random.normal(0, 0.5)
        
        prev_error = error
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'SV': sv,
        'PV': pv,
        'MV': mv
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data generated: {output_path}")
    print(f"Data points: {len(df)}")
    print(f"Duration: {duration}s")
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    generate_sample_data()
