from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map

from src.replay import plots_utils
from src.utils import conf_utils

conf = conf_utils.get_default_conf()


class Replay:
    def __init__(self, record_path, output_dir):
        self.output_dir = os.path.join(output_dir, os.path.basename(record_path).replace(".npz", ""))
        print(f"Output directory: {self.output_dir}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if os.path.islink(record_path):
            record_path = os.readlink(record_path)
        data = np.load(record_path)

        self.observation = np.array(data["observation"])
        self.realised_action = np.array(data["realised_action"])
        self.reward = np.array(data["reward"])

        self.pose = np.array(data["pose"])
        self.speed = np.array(data["speed"])
        self.wheels_omegas = np.array(data["wheels_omegas"])
        self.wheels_omegas_std = np.std(self.wheels_omegas, axis=1)
        self.steering_joint_angle = np.array(data["steering_joint_angle"])
        self.angular_velocity = np.array(data["angular_velocity"])
        self.wheel_poses = np.array(data["wheel_poses"])
        self.car_center = np.mean(self.wheel_poses, axis=1)

        self.track = np.array(data["track"])

        self.desired_speed = np.array(data["desired_speed"])
        self.curvature = np.array(data["curvature"])
        self.cte_control = np.array(data["cte_control"])
        self.he_control = np.array(data["he_control"])
        self.speed_error = np.array(data["speed_error"])
        self.cte = np.array(data["cte"])
        self.he = np.array(data["he"])
        self.driver = np.array(data["driver"])
        self.decision_curvature = np.array(data["decision_curvature"])

        self.steering_prediction = np.array(data["steering_prediction"]).reshape(-1)
        self.acceleration_prediction = np.array(data["acceleration_prediction"]).reshape(-1)

        self.teacher_action = np.array(data["teacher_action"])
        self.student_action = np.array(data["student_action"])
        self.noisy_state = np.array(data["noisy_state"]).reshape(-1, 8)
        self.noisy_speed = self.noisy_state[:, 0].reshape(-1) * 100
        self.noisy_wheels_omegas = self.noisy_state[:, 1:5] * 200
        self.noisy_wheels_omegas_std = np.std(self.noisy_wheels_omegas, axis=1)
        self.noisy_angular_velocity = -(self.noisy_state[:, 6].reshape(-1) * 3)
        self.noisy_steering_joint_angle = -(self.noisy_state[:, 7].reshape(-1) * 0.5)

        if "road_mask_prediction" in data:
            self.road_mask_prediction = np.array(data["road_mask_prediction"])
        if "chevron_mask_prediction" in data:
            self.chevron_mask_prediction = np.array(object=data["chevron_mask_prediction"])
        if "curvature_prediction" in data:
            self.curvature_prediction = (
                conf.CURVATURE_STD * np.array(data["curvature_prediction"]).flatten() + conf.CURVATURE_MEAN
            )

    def scatter_plot_inputs(self):
        # Scatter noisy and noiseless sensor data
        # speed, wheels_omegas_std, steering joint angle, angular velocity
        _, axs = plt.subplots(2, 4, figsize=(24, 12))
        axs[0, 0].scatter(self.speed, self.noisy_speed, label="Speed")
        axs[0, 0].set_xlabel("Noiseless Speed")
        axs[0, 0].set_ylabel("Noisy Speed")
        axs[0, 0].legend()

        axs[0, 1].scatter(self.wheels_omegas_std, self.noisy_wheels_omegas_std, label="Wheels Omega Std")
        axs[0, 1].set_xlabel("Noiseless WO Std")
        axs[0, 1].set_ylabel("Noisy WO Std")
        axs[0, 1].legend()

        axs[1, 0].scatter(self.steering_joint_angle, self.noisy_steering_joint_angle, label="Steering Joint Angle")
        axs[1, 0].set_xlabel("Noiseless Steering Joint Angle")
        axs[1, 0].set_ylabel("Noisy Steering Joint Angle")
        axs[1, 0].legend()

        axs[1, 1].scatter(self.angular_velocity, self.noisy_angular_velocity, label="Angular Velocity")
        axs[1, 1].set_xlabel("Noiseless Angular Velocity")
        axs[1, 1].set_ylabel("Noisy Angular Velocity")
        axs[1, 1].legend()

        # Scatter noiseless wheels omegas against noisy wheels omegas
        axs[0, 2].scatter(self.wheels_omegas[:, 0], self.noisy_wheels_omegas[:, 0], label="Wheel Omega 0")
        axs[0, 2].set_xlabel("Noiseless WO0")
        axs[0, 2].set_ylabel("Noisy WO0")
        axs[0, 2].legend()

        axs[0, 3].scatter(self.wheels_omegas[:, 1], self.noisy_wheels_omegas[:, 1], label="Wheel Omega 1")
        axs[0, 3].set_xlabel("Noiseless WO1")
        axs[0, 3].set_ylabel("Noisy WO1")
        axs[0, 3].legend()

        axs[1, 2].scatter(self.wheels_omegas[:, 2], self.noisy_wheels_omegas[:, 2], label="Wheel Omega 2")
        axs[1, 2].set_xlabel("Noiseless WO2")
        axs[1, 2].set_ylabel("Noisy WO2")
        axs[1, 2].legend()

        axs[1, 3].scatter(self.wheels_omegas[:, 3], self.noisy_wheels_omegas[:, 3], label="Wheel Omega 3")
        axs[1, 3].set_xlabel("Noiseless WO3")
        axs[1, 3].set_ylabel("Noisy WO3")
        axs[1, 3].legend()

        plt.savefig(os.path.join(self.output_dir, "input_scatter_plot.png"))

    def scatter_plot_outputs(self):
        _, axs = plt.subplots(2, ncols=2, figsize=(16, 16))
        if "curvature_prediction" in self.__dict__:
            axs[0, 0].scatter(self.curvature_prediction, self.curvature, label="Curvature")
            axs[0, 0].set_xlabel("Groundtruth Curvature")
            axs[0, 0].set_ylabel("Predicted Curvature")
            axs[0, 0].legend()

        axs[0, 1].scatter(self.teacher_action[:, 0], self.student_action[:, 0], label="Steering")
        axs[0, 1].set_xlabel("Teacher's steering")
        axs[0, 1].set_ylabel("Student's steering")
        axs[0, 1].legend()

        axs[1, 0].scatter(self.teacher_action[:, 1], self.student_action[:, 1], label="Gas")
        axs[1, 0].set_xlabel("Teacher's gas")
        axs[1, 0].set_ylabel("Student's gas")
        axs[1, 0].legend()

        axs[1, 1].scatter(self.teacher_action[:, 2], self.student_action[:, 2], label="Brake")
        axs[1, 1].set_xlabel("Teacher's brake")
        axs[1, 1].set_ylabel("Student's brake")
        axs[1, 1].legend()

        plt.savefig(os.path.join(self.output_dir, "output_scatter_plot.png"))

    def histogram_outputs(self):
        _, axs = plt.subplots(2, ncols=2, figsize=(16, 16))
        if "curvature_prediction" in self.__dict__:
            axs[0, 0].hist(self.curvature, label="Groundtruth Curvature", bins=50, alpha=0.5)
            axs[0, 0].hist(
                self.curvature_prediction,
                label="Predicted Curvature",
                bins=50,
                alpha=0.5,
            )
            axs[0, 0].legend()

        axs[0, 1].hist(self.teacher_action[:, 0], label="Groundtruth Steering", bins=50, alpha=0.5)
        axs[0, 1].hist(self.student_action[:, 0], label="Predicted Steering", bins=50, alpha=0.5)
        axs[0, 1].legend()

        axs[1, 0].hist(self.teacher_action[:, 1], label="Groundtruth Gas", bins=50, alpha=0.5)
        axs[1, 0].hist(self.student_action[:, 1], label="Predicted Gas", bins=50, alpha=0.5)
        axs[1, 0].legend()

        axs[1, 1].hist(self.teacher_action[:, 2], label="Groundtruth Brake", bins=50, alpha=0.5)
        axs[1, 1].hist(self.student_action[:, 2], label="Predicted Brake", bins=50, alpha=0.5)
        axs[1, 1].legend()

        plt.savefig(os.path.join(self.output_dir, "output_histogram.png"))

    def plot_all_frames(self):
        process_map(self._plot_frame, range(len(self.observation)), max_workers=conf.GYM_MAX_WORKERS)

    def _plot_frame(self, frame_id):
        self.car_center = np.mean(self.wheel_poses[frame_id], axis=0)
        # Plot the frame
        self.fig, self.axs = plt.subplots(
            2, ncols=3, figsize=(16, 16)
        )  # Create 3 subplots: 1 for the track, 1 for the data, and 1 for the observation
        self._plot_abstract_track(frame_id)
        self._plot_expert_control(frame_id)
        self._plot_observation(frame_id)
        self._plot_imitator_predictions(frame_id)
        self._plot_road_and_chevron_mask_prediction(frame_id)
        self._save_plot(frame_id)

    def _plot_abstract_track(self, frame_id):
        # First subplot: Plot the track and car's positions with orientation
        self.axs[0, 0].axis("off")
        self.axs[0, 0].plot(self.track[:, 0], self.track[:, 1], label="Track", color="gray")
        # axs[0, 0].scatter(pose[0], pose[1], color="red", label="Current Position")
        self.axs[0, 0].quiver(
            self.pose[frame_id][0],
            self.pose[frame_id][1],
            *plots_utils.get_car_orientation_arrow(self.pose[frame_id]),
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color="red",
            label="Ego Orientation",
        )  # Plot orientation arrows
        self.axs[0, 0].quiver(
            self.pose[frame_id][0],
            self.pose[frame_id][1],
            *plots_utils.get_action_arrow(self.realised_action[frame_id], self.pose[frame_id]),
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color="blue",
            label="Ego Action",
        )  # Plot the action arrow
        self.axs[0, 0].set_xlabel("X Position")
        self.axs[0, 0].set_ylabel("Y Position")
        self.axs[0, 0].legend()
        self.axs[0, 0].set_aspect("equal", adjustable="box")

    def _plot_expert_control(self, frame_id):
        # Second subplot: Display data in text format (debugging information)
        self.axs[0, 1].axis("off")
        debug_text = f"""
Expert's Inputs
    Speed: {self.speed[frame_id]:.2f}
    Wheel Velocity 0: {self.wheels_omegas[frame_id][0]:.2f}
    Wheel Velocity 1: {self.wheels_omegas[frame_id][1]:.2f}
    Wheel Velocity 2: {self.wheels_omegas[frame_id][2]:.2f}
    Wheel Velocity 3: {self.wheels_omegas[frame_id][3]:.2f}
    Wheel Velocity Std: {self.wheels_omegas_std[frame_id]:.2f}
    Angular velocity: {self.angular_velocity[frame_id]:.2f}
    Steering Joint Angle: {self.steering_joint_angle[frame_id]:.2f}

Expert's Intermediate Outputs
    Groundtruth Curvature: {self.curvature[frame_id]:.2f}
    Groundtruth Cross Track Error: {self.cte[frame_id]:.2f}
    Groundtruth Heading Error: {self.he[frame_id]:.2f}
    Groundtruth Desired Speed: {self.desired_speed[frame_id]:.2f}
    Groundtruth Speed Error: {self.speed_error[frame_id]:.2f}

Expert's Controls
    Groundtruth Steering: {self.teacher_action[frame_id][0]:.2f}
    Groundtruth Gas: {self.teacher_action[frame_id][1]:.2f}
    Groundtruth Brake: {self.teacher_action[frame_id][2]:.2f}
        """
        self.axs[0, 1].text(0.1, 0.5, debug_text, fontsize=12, horizontalalignment="left", verticalalignment="center")

    def _plot_imitator_predictions(self, frame_id):
        # Second subplot: Display data in text format (debugging information)
        self.axs[0, 2].axis("off")
        if "curvature_prediction" in self.__dict__:
            predicted_curvature_text = f"{self.curvature_prediction[frame_id]:.2f}"
        else:
            predicted_curvature_text = "-"
        debug_text = f"""
Imitator's Inputs
    Noisy Speed: {self.noisy_speed[frame_id]:.2f}
    Noisy Wheel Velocity 0: {self.noisy_wheels_omegas[frame_id][0]:.2f}
    Noisy Wheel Velocity 1: {self.noisy_wheels_omegas[frame_id][1]:.2f}
    Noisy Wheel Velocity 2: {self.noisy_wheels_omegas[frame_id][2]:.2f}
    Noisy Wheel Velocity 3: {self.noisy_wheels_omegas[frame_id][3]:.2f}
    Noisy Wheel Velocity Std: {self.noisy_wheels_omegas_std[frame_id]:.2f}
    Noisy Angular Velocity: {self.noisy_angular_velocity[frame_id]:.2f}
    Noisy Steering Joint Angle: {self.noisy_steering_joint_angle[frame_id]:.2f}

Imitator's Intermediate Outputs
    Predicted Curvature: {predicted_curvature_text}
    Predicted Cross Track Error: -
    Predicted Heading Error: -
    Predicted Desired Speed: -
    Predicted Speed Error: -

Imitator's Controls
    Predicted Steering: {self.student_action[frame_id][0]:.2f}
    Predicted Gas: {self.student_action[frame_id][1]:.2f}
    Predicted Brake: {self.student_action[frame_id][2]:.2f}
        """
        self.axs[0, 2].text(0.1, 0.5, debug_text, fontsize=12, horizontalalignment="left", verticalalignment="center")

    def _plot_observation(self, frame_id):
        # Third plot observation
        self.axs[1, 0].axis("off")
        self.axs[1, 0].imshow(self.observation[frame_id])
        self.axs[1, 0].set_title("Model's Input")

    def _plot_road_and_chevron_mask_prediction(self, frame_id):
        self.axs[1, 1].axis("off")
        if "road_mask_prediction" in self.__dict__:
            road_mask = F.sigmoid(torch.from_numpy(self.road_mask_prediction[frame_id].squeeze()))
            padded_road_mask = torch.zeros(96, 96)
            padded_road_mask[: road_mask.shape[0], : road_mask.shape[1]] = road_mask
            self.axs[1, 1].imshow(padded_road_mask)
            self.axs[1, 1].set_title("Road Mask Prediction")
        self.axs[1, 2].axis("off")
        if "chevron_mask_prediction" in self.__dict__:
            chevrons_mask = F.sigmoid(torch.from_numpy(self.chevron_mask_prediction[frame_id].squeeze()))
            padded_chevrons_mask = torch.zeros(96, 96)
            padded_chevrons_mask[: chevrons_mask.shape[0], : chevrons_mask.shape[1]] = chevrons_mask
            padded_chevrons_mask[:, -1] = 0  # Remove the last column because of padding
            self.axs[1, 2].imshow(padded_chevrons_mask)
            self.axs[1, 2].set_title("Chevron Mask Prediction")

    def _save_plot(self, frame_id):
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f"pose_{frame_id:03d}.png"), bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # argparse
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--record_path", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--plot_all_frames", action="store_true", help="Plot all frames (default: False)")
    args = args.parse_args(sys.argv[1:])
    replay = Replay(args.record_path, args.output_dir)
    if not bool(args.plot_all_frames):
        replay.scatter_plot_inputs()
        replay.scatter_plot_outputs()
        replay.histogram_outputs()
    else:
        replay.plot_all_frames()
