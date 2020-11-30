import json
import jsonmerge
import random
import datetime
from pathlib import Path
from os.path import join

from lfh.utils.io import create_output_dir
from lfh.utils.io import read_json

_type_conversion_table = {
    "None": lambda _: None,
    "float": lambda x: float(x),
    "int": lambda x: int(x),
    "str": lambda x: str(x),
    "bool": lambda x: x.lower() == "true",
    "list": lambda x: list(x)
}


def convert_str_to_type(value, type_str):
    if type_str in _type_conversion_table:
        return _type_conversion_table[type_str](value)
    else:
        return None


class Configurations:

    def __init__(self, params, note):
        """D: establish of configuration-related and logging directory stuff.
        Main changes from Allen's code: save in hard disk rather than precious
        SSD space, added date string and remove annoying parentheses.
        """
        self.global_config = read_json(dir_path=params.exp_name,
                                       file_name="global_settings")
        self.exp_config = read_json(dir_path=params.exp_name, file_name=params.exp_id)
        self.exp_name = "_{}".format(params.exp_id)
        self.profile = params.profile

        # set random seed
        if params.seed is not None:
            self.exp_config["seed"] = params.seed
        elif "seed" not in self.exp_config:
            self.exp_config["seed"] = random.randint(10, 100000)


        self.exp_config["agent"] = params.agent

        if params.agent == "uniform_zpd" or params.agent == "unseq_DDQN":
            self.exp_config["zpd"] = {}
            self.exp_config["zpd"]["offset"] = params.offset
            self.exp_config["zpd"]["radius"] = params.radius
            self.exp_config["zpd"]["mix_ratio"] = params.mix_ratio
            self.exp_config["zpd"]["demonstrations_dir"] =  params.demonstrations_dir


        self.note = note
        self.params = jsonmerge.merge(
            base=self.global_config,
            head=self.exp_config)

        self.params["exp"] = self.exp_name

        if len(note) > 0:
            note = "_" + note.replace(" ", "_").lower()

        # D: extra stuff for more scalable logging, also check if teaching.
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        last = "{}{}_{}_s{}".format(self.params["exp"], note, date,
                                    self.params["seed"])

        
        self.params["log"]["dir"] = str(Path( self.params["log"]["root"])/last)
        # __import__('pdb').set_trace()
        create_output_dir(params=self.params)

    def merge_keys(self, merge_key):
        self.params[merge_key] = {}
        for section_key, section in self.params.items():
            if merge_key in section:
                self.params[merge_key] = {
                    **self.params[merge_key],
                    **section[merge_key]}

    def dump(self, filename="params.txt"):
        with open(join(self.params["log"]["dir"], filename), 'w') as f:
            json.dump(self.params, f, sort_keys=True, indent=4)
            f.write('\n')
        return self.params
