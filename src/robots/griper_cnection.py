import mujoco


def create_robot(robot_path, gripper_path):
    robot = mujoco.MjSpec.from_file(str(robot_path))
    gripper = mujoco.MjSpec.from_file(str(gripper_path))

    robot.attach(
        gripper,
        prefix="gripper_",
        site=robot.site("attachment_site"),
    )

    model = robot.compile()
    data = mujoco.MjData(model)

    return model, data