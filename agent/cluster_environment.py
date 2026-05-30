import random


class ClusterEnvironment:
    def __init__(self):
        # The hidden variable: fluctuating baseline user requests
        self._traffic_load = 5000.0
        self.iteration_count = 0

    def set_instance_count(self, instances: int) -> dict:
        """
        Accepts the number of server instances to provision.
        Returns the resulting average CPU utilization percentage of the cluster.
        """
        self.iteration_count += 1
        if instances <= 0:
            return {"cpu_utilization": 100.0, "status": "CRASHED - No instances"}

        # Slight traffic fluctuation over time to simulate a real environment
        self._traffic_load += random.uniform(-150, 150)

        # CPU utilization formula (Hidden from the agent)
        # More instances = lower CPU per instance.
        # Crucially, it's non-linear due to clustering overhead.
        base_cpu = (self._traffic_load / (instances ** 1.2))

        # Cap CPU between 1% and 100%
        cpu_utilization = max(1.0, min(100.0, base_cpu))

        return {
            "cpu_utilization": round(cpu_utilization, 2),
            "iteration": self.iteration_count,
        }
