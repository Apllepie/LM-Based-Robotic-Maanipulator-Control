import mujoco



def create_robot(robot_path, gripper_path):
    robot = mujoco.MjSpec.from_file(str(robot_path))
    gripper = mujoco.MjSpec.from_file(str(gripper_path))

    robot.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    robot.option.impratio = 10
    robot.attach(
        gripper,
        prefix="gripper_",
        site=robot.site("attachment_site"),
    )

    model = robot.compile()
    data = mujoco.MjData(model)

    gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_fingers_actuator",)

    print("Gripper actuator ID:", gripper_id)

    return model, data