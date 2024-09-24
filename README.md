<h1 align="center">
    Solving OpenAI Gym CarRace-v2 with Imitation Learning
</h1>

<p align="center">
  <img src="static/single_task.png" alt="Single-task learning baseline" width="400"/>
</p>

## Table of content
- [Table of content](#table-of-content)
- [Results 🔥](#results-)
  - [Original Challenge within 1000 Steps](#original-challenge-within-1000-steps)
    - [Experts](#experts)
    - [End-To-End Models](#end-to-end-models)
  - [Short Challenge within 600 Steps](#short-challenge-within-600-steps)
    - [Experts](#experts-1)
    - [End-To-End Models](#end-to-end-models-1)
- [Demo 🔥](#demo-)



## Results 🔥

<b>In this setting experts have all privileged accesses to a noise-free world map, vehicle's pose and state. Imitators have only accesses to a noisy bird-eye-view and state of the car.</b>.

Following expert and imitative drivers are available:

- Path-Following experts:
  - Linear PID controller.
  - Geometric controllers: Pure Pursuit & Stanley controller.
- Imitatitive end-to-end learning models:
  - Single-task CNN model with only control prediction heads.
  - Multi-task learning CNN architecture for regularized representation learning.

Benchmarks were run on 100 random fixed seeds. CarRacing-v0 defines "solving" as getting average reward of 900 over 100 consecutive trials, each trial runs at most 1000 steps.

### Original Challenge within 1000 Steps

#### Experts

<table align="center">
    <tr>
        <th>Experts</th>
        <th>Mean</th>
        <th>Min</th>
        <th>Max</th>
        <th>Challenge Solved</th>
    </tr>
    <tr>
        <td>PID</td>
        <td>920 &plusmn; 27</td>
        <td>778</td>
        <td>944</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Pure Pursuit</td>
        <td>896 &plusmn; 29</td>
        <td>815</td>
        <td>945</td>
        <td>❌</td>
    </tr>
    <tr>
        <td>Stanley</td>
        <td>882 &plusmn; 37</td>
        <td>732</td>
        <td>944</td>
        <td>❌</td>
    </tr>
</table>
<br>

#### End-To-End Models

<table align="center">
    <tr>
        <th>Architecture</th>
        <th>Mean</th>
        <th>Min</th>
        <th>Max</th>
        <th>Expert</th>
        <th>% Expert</th>
        <th>% Solved</th>
        <th>Challenge Solved</th>
    </tr>
    <tr>
        <td>Single-Task CNN</td>
        <td>895 &plusmn; 103</td>
        <td>219</td>
        <td>949</td>
        <td>PID</td>
        <td>97%</td>
        <td>99%</td>
        <td>❌</td>
    </tr>
    <tr>
        <td>Multi-Task CNN</td>
        <td>846 &plusmn; 173</td>
        <td>204</td>
        <td>948</td>
        <td>PID</td>
        <td>91%</td>
        <td>94%</td>
        <td>❌</td>
    </tr>
</table>

### Short Challenge within 600 Steps

Original challenge from University Tübingen. The challenge is considered solved If the model achieved on average 700 points after 600 Steps in 100 trials.

#### Experts

<table align="center">
    <tr>
        <th>Experts</th>
        <th>Mean</th>
        <th>Min</th>
        <th>Max</th>
        <th>Challenge Solved</th>
    </tr>
    <tr>
        <td>PID</td>
        <td>801 &plusmn; 10</td>
        <td>536</td>
        <td>944</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Pure Pursuit</td>
        <td>744 &plusmn; 98</td>
        <td>533</td>
        <td>945</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Stanley</td>
        <td>737 &plusmn; 114</td>
        <td>391</td>
        <td>994</td>
        <td>✅</td>
    </tr>
</table>

#### End-To-End Models

<table align="center">
    <tr>
        <th>Architecture</th>
        <th>Mean</th>
        <th>Min</th>
        <th>Max</th>
        <th>Expert</th>
        <th>% of Expert</th>
        <th>Challenge Solved</th>
    </tr>
    <tr>
        <td>Single-Task CNN</td>
        <td>801 &plusmn; 125</td>
        <td>259</td>
        <td>949</td>
        <td>PID</td>
        <td>100%</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Multi-Task CNN</td>
        <td>753 &plusmn; 180</td>
        <td>244</td>
        <td>948</td>
        <td>PID</td>
        <td>94%</td>
        <td>✅</td>
    </tr>
</table>

## Demo 🔥

Download the best trained weights for a task

<table align="center">
    <tr>
        <th>Architecture</th>
        <th>Link</th>
        <th>Task</th>
    </tr>
    <tr>
        <td>Single-Task CNN</td>
        <td><a href="https://api.wandb.ai/artifactsV2/default/long-pollehn1/QXJ0aWZhY3Q6MTI0NjQ5NTM1Mg%3D%3D/5e50f56ec06e4a577644ec383988b733/2024_09_24_cgpool1905_12_33_58_SingleTaskCNN135_824.pth">2024_09_24_cgpool1905_12_33_58_SingleTaskCNN135_824.pth</a></td>
        <td>1000 Steps</td>
    </tr>
    <tr>
        <td>Single-Task CNN</td>
        <td><a href="https://api.wandb.ai/artifactsV2/default/long-pollehn1/QXJ0aWZhY3Q6MTIzNzkyNTQ1Nw%3D%3D/94050699f17a1171a1b1a3b4e470ebba/2024_09_20_cgpool1902_10_10_01_SingleTaskCNN125_822.pth">2024_09_20_cgpool1902_10_10_01_SingleTaskCNN125_822.pth</a></td>
        <td>600 Steps</td>
    </tr>
</table>

To drive with the model

```bash
python3 -m src.demo <path_to_model>
```

To benchmark

```bash
python3 -m src.record --student_controller imitation --student_model_path <path_to_model> --teacher_controller pid --max_steps 1000
```

For controller debugging

```bash
python3 -m src.replay.replay --record_path <path> --outputp_dir <path> --plot_all_frames
```

The last script will produce a replay of the model comparing with ground truth expert's decisions.

<p align="center">
  <img src="static/output.gif" alt="Dagger" width="675"/>
</p>
