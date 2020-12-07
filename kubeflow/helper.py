import kfp
import logging
import multiprocessing
import uuid
import datetime
from typing import Dict, List

def my_config_init(self, host="http://localhost",
    api_key=None, api_key_prefix=None,
    username=None, password=None,
    discard_unknown_keys=False,
    ):
    """Constructor
    """
    self.host = host
    """Default Base url
    """
    self.temp_folder_path = None
    """Temp file folder for downloading files
    """
    # Authentication Settings
    self.api_key = {}
    if api_key:
        self.api_key = api_key
    """dict to store API key(s)
    """
    self.api_key_prefix = {}
    if api_key_prefix:
        self.api_key_prefix = api_key_prefix
    """dict to store API prefix (e.g. Bearer)
    """
    self.refresh_api_key_hook = None
    """function hook to refresh API key if expired
    """
    self.username = username
    """Username for HTTP basic authentication
    """
    self.password = password
    """Password for HTTP basic authentication
    """
    self.discard_unknown_keys = discard_unknown_keys
    self.logger = {}
    """Logging Settings
    """
    self.logger["package_logger"] = logging.getLogger("kfp_server_api")
    self.logger["urllib3_logger"] = logging.getLogger("urllib3")
    self.logger_format = '%(asctime)s %(levelname)s %(message)s'
    """Log format
    """
    self.logger_stream_handler = None
    """Log stream handler
    """
    self.logger_file_handler = None
    """Log file handler
    """
    self.logger_file = None
    """Debug file location
    """
    self.debug = False
    """Debug switch
    """

    self.verify_ssl = False
    """SSL/TLS verification
        Set this to false to skip verifying SSL certificate when calling API
        from https server.
    """
    self.ssl_ca_cert = None
    """Set this to customize the certificate file to verify the peer.
    """
    self.cert_file = None
    """client certificate file
    """
    self.key_file = None
    """client key file
    """
    self.assert_hostname = None
    """Set this to True/False to enable/disable SSL hostname verification.
    """

    self.connection_pool_maxsize = multiprocessing.cpu_count() * 5
    """urllib3 connection pool's maximum number of connections saved
        per pool. urllib3 uses 1 connection as default value, but this is
        not the best value when you are making a lot of possibly parallel
        requests to the same host, which is often the case here.
        cpu_count * 5 is used as default value to increase performance.
    """

    self.proxy = None
    """Proxy URL
    """
    self.proxy_headers = None
    """Proxy headers
    """
    self.safe_chars_for_path_param = ''
    """Safe chars for path_param
    """
    self.retries = None
    """Adding retries to override urllib3 default value 3
    """
    # Disable client side validation
    self.client_side_validation = True

import splogger
import warnings
from kfp_server_api.configuration import Configuration
from kubernetes.client.models.v1_volume import V1Volume
from kubernetes.client.models.v1_affinity import V1Affinity
from kubernetes.client.models.v1_node_affinity import V1NodeAffinity
from kubernetes.client.models.v1_node_selector import V1NodeSelector
from kubernetes.client.models.v1_node_selector_requirement import V1NodeSelectorRequirement
from kubernetes.client.models.v1_node_selector_term import V1NodeSelectorTerm
from kubernetes.client.models.v1_host_path_volume_source import V1HostPathVolumeSource
import uuid

Configuration.__init__  = my_config_init


# HELPERS

class DVICContainerOperation:
    """
        Wrapper for a container operation on the Kubeflow DVIC

        image: the docker image to execute
        args: arguments passed to the image at execution time
        file_outputs: Dict[str:str] of output files to go to the ARTIFACTS
    """

    def __init__(self, image : str, *args : List[str], name : str = "unnamed-operation", file_outputs : Dict[str,str] = None):
        # file_outputs={"hello_output": "/out"} 
        # See https://www.kubeflow.org/docs/pipelines/sdk/pipelines-metrics/ for metrics
        self.image = image
        self.args = args
        self.name = name
        self.file_outputs = file_outputs
        self.node = None
        self.ngpu = None
        self.volumes = []
        self.after = []
        self.wdir = None
        if DVICPipelineWrapper.current_pipeline:
            DVICPipelineWrapper.current_pipeline += self

    def __call__(self):
        """
        To be called in a pipeline
        """
        self.op = kfp.dsl.ContainerOp(self.name, self.image, arguments=self.args, file_outputs = self.file_outputs)
        if self.node:
            self.op.add_affinity(V1Affinity(node_affinity=V1NodeAffinity(required_during_scheduling_ignored_during_execution=V1NodeSelector(node_selector_terms=[V1NodeSelectorTerm(match_expressions=[V1NodeSelectorRequirement(key='kubernetes.io/hostname', operator='In', values=[self.node])])]))))
        if self.ngpu:
            self.op.set_gpu_limit(self.ngpu)
        if self.working_dir:
            self.op.working_dir = self.wdir
        if len(self.volumes) > 0:
            for vol in self.volumes:
                self.op.add_pvolumes(vol)
    
    def _apply_order(self):
        if len(self.after) > 0:
            for e in self.after:
                self.op.after(e.op)

            
    def __or__(self, other):
        other.after.append(self)
        return other

    def select_node(self, name="dgx.dvic.devinci.fr"):
        """
        Force execution on a specific node, the dgx by default
        """
        self.node = name
        return self

    def gpu(self, request=1):
        """
        Requested number of GPUs
        """
        self.ngpu = request
        return self

    def working_dir(self, path):
        self.wdir = path
        return self

    def mount_host_path(self, container_path, host_path, name=None):
        """
        Mount the host_path path in the container at container_path
        """
        if not name:
            name = str(uuid.uuid4())
        self.volumes.append({container_path: V1Volume(name=name, host_path=V1HostPathVolumeSource(host_path))})
        return self


class DVICPipelineWrapper:

    current_pipeline = None

    def __init__(self, name, description, exp = None, namespace="dvic-kf"):
        self.name = name
        self.description = description
        self.namepsace = namespace
        self.elems = []
        self.exp = name if not exp else exp
        self.func = None
        self.res = None

    def set_exp(self, exp):
        self.exp = exp
        return self

    def __enter__(self):
        splogger.fine("Building pipeline")
        DVICPipelineWrapper.current_pipeline = self
        return self
    
    def __exit__(self, ex, exc , texc):
        pass

    def __iadd__(self, elem: DVICContainerOperation):
        self.elems.append(elem)
        return self

    def set_func(self, func):
        @kfp.dsl.pipeline(
            name=self.name,
            description=self.description
        )
        def _wrapper():
           func()
        self.func = _wrapper

    def _generic_pipeline(self):
        @kfp.dsl.pipeline(
            name=self.name,
            description=self.description
        )
        def _wrapper():
            for operation in self.elems:
                operation()
            for operation in self.elems:
                operation._apply_order()
        return _wrapper

    def wait(self):
        if not self.res:
            return self
        self.res.wait_for_run_completion()

    def __call__(self, **kwargs):
        """
            Run the pipeline
        """
        splogger.fine(f'Starting pipeline {self.name}')
        warnings.filterwarnings("ignore")
        c = kfp.Client("https://kubflow.dvic.devinci.fr/pipeline")
        self.res = c.create_run_from_pipeline_func(self.func if self.func != None else self._generic_pipeline(), kwargs, f'{self.name} {str(datetime.datetime.now())}', self.name, namespace=self.namepsace)
        splogger.success(f'Pipeline started', strong=True)
        splogger.success(f'Pipeline URL: https://kubflow.dvic.devinci.fr/_/pipeline/#/experiments/details/{self.res.run_id}')
        return self

# END HELPERS

# @kfp.dsl.pipeline(
#     name="Test pipeline2",
#     description="Testing KF"
# )
# def generic_pipeline():
#     # file_outputs={"hello_output": "/out"}
#     container_element("test_element", "win32gg/testw").
#     op = kfp.dsl.ContainerOp("generer_dataset", "win32gg/testw", arguments=("-p", "/data/out"))
#     select_node(op, 'dgx.dvic.devinci.fr')
#     mount_host_path(op, '/data', 'dgx-data', '/data/dl/test')

# c = kfp.Client("https://kubflow.dvic.devinci.fr/pipeline")
# c.create_run_from_pipeline_func(generic_pipeline, {}, "test_pipeline", "test", namespace="kubeflow-dvic")

#  ==== EXAMPLE WITH HELPER ====

if __name__ == '__main__':

    with DVICPipelineWrapper("my-pipeline", "This is my pipeline").set_exp("blue-example") as pipeline:

        # Elements
        elem1 = DVICContainerOperation("win32gg/testw", "-p", "/data/out").select_node().mount_host_path("/data", "/data/dl/test")
        elem2 = DVICContainerOperation("win32gg/testw", "-p", "/data/oue").select_node().mount_host_path("/data", "/data/dl/test")

        # Execution order, elem1 then elem2
        elem1 | elem2

        # Start pipeline
        pipeline()
