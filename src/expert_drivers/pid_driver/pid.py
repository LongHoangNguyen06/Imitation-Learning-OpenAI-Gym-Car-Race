from __future__ import annotations


class PIDController:
    def __init__(self, kp, ki, kd):
        """
        Initialize the PID controller without a setpoint.

        Args:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        ki (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.p = 0
        self.i = 0
        self.d = 0
        self.derivative = 0

    def update(self, error):
        """
        Update the PID controller with the given error and compute the control output.

        Args:
        dt (float): The time difference between the current and previous update.

        Returns:
        control (float): The computed control output.
        """
        # Proportional term
        self.p = self.kp * error

        # Integral term
        self.integral += error
        self.i = self.ki * self.integral

        # Derivative term
        self.derivative = error - self.previous_error
        self.d = self.kd * self.derivative

        # Update previous error
        self.previous_error = error

        # Compute the control output
        return self.p + self.i + self.d

    def reset(self):
        """
        Reset the integral and previous error values (useful to call when you restart the controller).
        """
        self.integral = 0
        self.previous_error = 0
        self.p = 0
        self.i = 0
        self.d = 0
        self.derivative = 0
