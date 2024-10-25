import numpy as np
from models.model import Model
from controllers.controller import Controller
import pandas as pd  # Για αποθήκευση σε CSV


class ModelController(Controller):
    def __init__(
            self,
            model: Model,
            ctrl_limits: list[np.ndarray],
            num_actuators: int,
            one_hot_signs: dict[str, np.ndarray],
            one_hot_orders: dict[int, np.ndarray]
    ):
        super().__init__(ctrl_limits=ctrl_limits)

        self._model = model
        self._num_actuators = num_actuators
        self._one_hot_signs = one_hot_signs
        self._one_hot_orders = one_hot_orders

        self._sign_vector = None
        self._is_done = False
        self._transition_counter = 0

    def _set_sign(self, sign: str):
        self._sign_vector = self._one_hot_signs[sign]

    def _get_next_control(self, sign: str, order: int) -> np.ndarray or None:
        if order in self._one_hot_orders:
            next_ctrl = self._model.predict_next_control(
                sign_vector=self._sign_vector,
                order_vector=self._one_hot_orders[order]
            )

            assert next_ctrl.ndim == 1 and next_ctrl.shape[0] == self._num_actuators
            self.save_control_output(next_ctrl)
        else:
            next_ctrl = None
        return next_ctrl

    def save_control_output(self, control_output: np.ndarray):
        # Δημιουργία DataFrame και αποθήκευση σε CSV
        df = pd.DataFrame(control_output.reshape(1, -1),
                          columns=[f'Actuator_{i + 1}' for i in range(self._num_actuators)])
        df.to_csv('control_output.csv', mode='a', header=False, index=False)