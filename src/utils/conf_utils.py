from __future__ import annotations

from dotmap import DotMap
from dynaconf import Dynaconf


def dynaconf_to_dotmap(conf: Dynaconf) -> DotMap:
    """
    Converts a Dynaconf object to a DotMap.

    Args:
        conf (Dynaconf): The Dynaconf object to convert.

    Returns:
        DotMap: A DotMap representation of the Dynaconf object.
    """
    # Convert the Dynaconf object to a dictionary and then to DotMap
    conf_dict = conf.to_dict()
    dotmap_conf = DotMap(conf_dict)

    return dotmap_conf


def get_conf(controller: str, print_out=False) -> Dynaconf:
    """
    Retrieves the configuration settings based on the given arguments.
    Args:
        args (Namespace): The command line arguments.
    Returns:
        Dynaconf: The configuration object.
    Raises:
        ValueError: If the mode or controller is invalid.
    """
    settings_files = ["src/conf/default_conf.py"]

    if controller == "human":
        settings_files.append("src/conf/controller_conf/human_conf.py")
    elif controller == "constant":
        settings_files.append("src/conf/controller_conf/constant_conf.py")
    elif controller == "pid":
        settings_files.append("src/conf/controller_conf/pid_conf.py")
    elif controller == "pure_pursuit":
        settings_files.append("src/conf/controller_conf/pure_pursuit_conf.py")
    elif controller == "stanley":
        settings_files.append("src/conf/controller_conf/stanley_conf.py")
    elif controller == "imitation":
        settings_files.append("src/conf/controller_conf/imitation_conf.py")
    else:
        raise ValueError(f"Invalid controller: {controller}")
    conf = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings_files, lowercase_read=False)
    if print_out:
        print("#" * 100)
        print("# Configuration")
        print("#" * 100)
        for key, value in conf.to_dict().items():
            print(f"{key}: {value}")
    return dynaconf_to_dotmap(conf)


def extend_conf(conf: Dynaconf, additional_config_files: list[str]) -> Dynaconf:
    """
    Extends the configuration settings with additional files.
    Args:
        conf (Dynaconf): The configuration object.
        additional_config_files (list[str]): The list of additional configuration files.
    Returns:
        Dynaconf: The extended configuration object.
    """
    conf = conf.copy()
    if additional_config_files is None:
        return conf
    for file in additional_config_files:
        new_conf = Dynaconf(settings_files=[file], lowercase_read=False)
        dotmap_new_conf = dynaconf_to_dotmap(new_conf)
        for key, value in dotmap_new_conf.items():
            conf[key] = value
    return conf


def get_default_conf():
    """
    Retrieve the default configuration settings.
    This function initializes and returns a Dynaconf object that loads
    settings from the specified configuration file.
    Returns:
        Dynaconf: An instance of Dynaconf with settings loaded from
        "src/conf/default_conf.py".
    """

    return dynaconf_to_dotmap(Dynaconf(settings_files=["src/conf/default_conf.py"]))
