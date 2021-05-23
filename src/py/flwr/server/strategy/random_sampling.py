"""
Random sampling of the clients
"""
from typing import Callable, Dict, Optional, Tuple, List

from flwr.common import Weights, Scalar, Parameters, FitIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .fedavg import FedAvg
from logging import INFO

WAIT_TIMEOUT = 600


class RandomSampling(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Block until `min_num_clients` are available
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        success = client_manager.wait_for(
            num_clients=min_num_clients, timeout=WAIT_TIMEOUT
        )

        if not success:
            # Do not continue if not enough clients are available
            log(
                INFO,
                "FedRandom: not enough clients available after timeout %s",
                WAIT_TIMEOUT,
            )
            return []

        clients = self._one_over_k_sampling(client_manager=client_manager)

        # Prepare parameters and config
        config = {}
        if self.on_fit_config_fn is not None:
            # Use custom fit config function if provided
            config = self.on_fit_config_fn(rnd)

        # TODO: This could be different
        config["timeout"] = 30

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def _one_over_k_sampling(
        self, client_manager: ClientManager
    ) -> List[ClientProxy]:
        """Sample clients with probability 1/k."""
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # `criterion` is also available for .sample(...)

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return clients
