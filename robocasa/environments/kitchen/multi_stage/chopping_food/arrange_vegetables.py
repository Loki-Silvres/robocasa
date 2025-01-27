from robocasa.environments.kitchen.kitchen import *


class ArrangeVegetables(Kitchen):
    """
    Arrange Vegetables: composite task for Chopping Food activity.

    Simulates the task of arranging vegetables on the cutting board.

    Steps:
        Take the vegetables from the sink and place them on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the vegetables from the sink and place them on the cutting board."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.45, 0.55)
                    ),
                    size=(0.35, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.45, 0.55)
                    ),
                    size=(0.45, 0.45),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            self, "vegetable1", "cutting_board"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            self, "vegetable2", "cutting_board"
        )
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="cutting_board")

        return (
            vegetable1_cutting_board_contact
            and vegetable2_cutting_board_contact
            and gripper_obj_far
        )
