<h1 align="center">
    Solving OpenAI Gym CarRace-v2 with Imitation Learning
</h1>

<p align="center">
  <img src="static/single_task.png" alt="Single-task learning baseline" width="400"/>
</p>

## Table of content
- [Table of content](#table-of-content)
- [Results 🔥](#results-)
  - [Original Challenge within 1000 Iterations](#original-challenge-within-1000-iterations)
    - [Experts](#experts)
    - [End-To-End Models](#end-to-end-models)
  - [Short Challenge within 600 Iterations](#short-challenge-within-600-iterations)
    - [Experts](#experts-1)
    - [End-To-End Models](#end-to-end-models-1)
- [Demo 🔥](#demo-)
- [Methods](#methods)
  - [PID Longitudinal Controller](#pid-longitudinal-controller)
  - [PID Lateral Controller](#pid-lateral-controller)
  - [Corner cutting](#corner-cutting)
  - [Pure Pursuit Lateral Controller](#pure-pursuit-lateral-controller)
  - [Stanley Lateral Controller](#stanley-lateral-controller)
  - [Single-Task CNN Imitator](#single-task-cnn-imitator)
  - [Multi-Task Learning CNN Imitator](#multi-task-learning-cnn-imitator)
  - [Data Aggregation](#data-aggregation)



## Results 🔥

<b>In this setting experts have all privileged accesses to a noise-free world map, vehicle's pose and state. Imitators have only accesses to a noisy bird-eye-view and state of the car.</b>.

Following expert and imitative drivers are available:

- Path-Following experts:
  - Linear PID controller.
  - Geometric controllers: Pure Pursuit & Stanley controller.
- Imitatitive end-to-end learning models:
  - Single-task CNN model with only control prediction heads.
  - Multi-task learning CNN architecture for regularized representation learning.

Benchmarks were run on 100 random fixed seeds. CarRacing-v0 defines "solving" as getting average reward of 900 over 100 consecutive trials, each trial runs at most 1000 iterations.

### Original Challenge within 1000 Iterations

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
        <td>883 &plusmn; 127</td>
        <td>219</td>
        <td>949</td>
        <td>PID</td>
        <td>95%</td>
        <td>98%</td>
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

### Short Challenge within 600 Iterations

Original challenge from University Tübingen. The challenge is considered solved If the model achieved on average 700 points after 600 iterations in 100 trials.

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

Download the best trained weights

```bash
wget https://api.wandb.ai/artifactsV2/default/long-pollehn1/QXJ0aWZhY3Q6MTIzNzkyNTQ1Nw%3D%3D/94050699f17a1171a1b1a3b4e470ebba/2024_09_20_cgpool1902_10_10_01_SingleTaskCNN125_822.pth
```

To run the models in a GUI

```bash
python3 -m src.demo 2024_09_20_cgpool1902_10_10_01_SingleTaskCNN125_822.pth
```

To benchmark and store the controller internal states

```bash
python3 -m src.record --student_controller imitation --student_model_path 2024_09_20_cgpool1902_10_10_01_SingleTaskCNN125_822.pth --max_iterations 600
```

For controller debugging

```bash
python3 -m src.replay.replay --record_path <path> --outputp_dir <path> --plot_all_frames
```

The last script will produce a replay of the model comparing with ground truth expert's decisions.

<p align="center">
  <img src="static/output.gif" alt="Dagger" width="675"/>
</p>


## Methods

### PID Longitudinal Controller

Define desired speed $S_d$ as linear interpolation between maximal speed $S_M$ and minimal speed $S_m$, scaled by curvature $c$ of the reference trajectory.

$$S_d = S_M - c\frac{(S_M - S_m)}{c_M}$$

Gas input to the car is the error between current speed and desired speed.

### PID Lateral Controller

One PID controller for cross track error and and one controller for the heading error between reference trajectory and car.

When the standard deviation of the four wheels' rotatory speed is too high, the desired speed will be reduced. This step is crucial to avoid oversteering.

### Corner cutting

One of the most important aspect is cutting corner. Here I found two strategies:

- The first one by choosing suitable reference trajectory to compute CTE and HE. By skipping intermediate way points when the road curvature is high, the corner cutting can be achieved.

- Smooth the path with the Chaik algorithm or spline.

### Pure Pursuit Lateral Controller

Given

- Lookahead Distance $L_d$.
- Current Position $x, y$.
- Target Point $x_t, y_t$.
- Heading Angle $\theta$.
- Steering Angle $\delta$.
- Wheel base $L$.

Using bicycle assumption to compute the steering correction $\delta$ as

$$\alpha = \arctan\left(\frac{y_t - y}{x_t - x}\right) - \theta$$

$$\delta = \arctan\left(\frac{2L \sin(\alpha)}{L_d}\right)$$

### Stanley Lateral Controller

The Stanley controller combines cross-track error (CTE) with heading error and is a bit more robust in the theory. Since I did not spend enough time into tuning the controllers, both Pure Pursuit and Stanley Controller seems to be equal good.

Note the PID, Pure Pursuit, and Stanley experts all use a PID controller for longitudinal control, differing only in their lateral steering methods.

### Single-Task CNN Imitator

This baseline network predicts only steering and gas outputs and consists of almost 190,000 parameters. Heuristical binary search was used to determine the optimal number of parameters.

### Multi-Task Learning CNN Imitator

Extends the baseline by using the backbone's features to predict:

- Road segmentation: Predicting a binary mask of the drivable area.
- Curvature: Estimation of the road's curvature.
- Existence of Chevron signs: Binary prediction of existence of chevron signs, which indicate sharp turns.

This architecture consists of about 500,000 parameters in training and have the same number of parameters as the Single-Task CNN at test time.

### Data Aggregation

Following code snippet presents the idea of DAgger.

```python
def dagger_iteration(student_driver, teacher_driver, env):
    history = defaultdict(list)

    done = False
    observation = env.reset()
    while not done:
        state = env.get_state()
        student_action = student_driver.get_action(observation, state)
        teacher_action = teacher_driver.get_action(observation, state)
        history["input"].append((observation, state))
         # Roll out randomly either student's action or teacher's action.
        observation, done = env.step(np.random.choice([teacher_action, student_action], [0.99**epoch, 1-0.99**epoch]))
        history["action"].append(teacher_action) # Store only teacher's action.
    return history

# Training loop
data = []
for _ in range(epochs):
    data += dagger_iteration(student_driver, teacher_driver, env)
    student_driver.learn(data) # Learn from data
```

To accelerate the training, the student model doesn't learn from all states. Some small tricks were employed to make DAgger faster. In each epoch, namely
- only most recent data and a portion of random states are used for gradient descent. Otherwise the training
time per epoch would increase quadratically.
- roll out with too high rewards will be discarded.
- states with higher losses have higher priority.

DAgger (red in the figure) over-performs simple behavior cloning (green in the figure) by a huge margin.

<p align="center">
  <img src="static/dagger.png" alt="Dagger" width="675"/>
</p>
