from abc import ABC, abstractmethod
from functools import partial
import datasets

# Abstract base class for all distinguishers
class Distinguisher(ABC):
    def __init__(self, llm, persona, origin_A, origin_B):
        self.llm = llm
        self.persona = persona
        self.origin_A = origin_A
        self.origin_B = origin_B

    def set_origin(self, origin_A, origin_B):
        self.origin_A = origin_A
        self.origin_B = origin_B
    
    @staticmethod
    @abstractmethod
    def prompt_fn(lm, persona, **kwargs):
        pass

    @staticmethod
    def flip_choices(output, prefix=""):
        # ok to override in derived clases
        choice = f"{prefix}choice"
        if output[choice] == "A":
            output[choice] = "B"
        elif output[choice] == "B":
            output[choice] = "A"
        else:
            # sad face model didnt work
            output[choice] == "FAIL"
            pass

    def apply_prompt(self, datarow, flip, prefix=""):
        # load input dictionary
        inputs = {k: datarow[k] for k in self.input_keys}
        inputs['A'] = self.origin_A
        inputs['B'] = self.origin_B
        if flip:
            inputs['A'], inputs['B'] = inputs['B'], inputs['A']
            prefix+="flipped_"

        # perform inference
        output = self.llm+self.prompt_fn(persona=self.persona, **inputs)
        output_dict = {prefix+k: output[k] for k in self.output_keys}

        # flip outputs if needed
        if flip:
            self.flip_choices(output_dict, prefix)

        return output_dict


    def distinguish_row(self, datarow, prefix=""):
        output = self.apply_prompt(datarow, flip=False, prefix=prefix)
        output.update(self.apply_prompt(datarow, flip=True, prefix=prefix))
        return output

    def distinguish(self, dataset, prefix=""):
        return dataset.map(
            partial(self.distinguish_row, 
                prefix=prefix)
        )
    
    def distinguish_majority(self, dataset, count, prefix=""):
        print(f"Running distinguisher on {dataset.num_rows} rows with a count of {count} for a total of {dataset.num_rows*count} trials")
        for i in range(count):
            dataset = self.distinguish(dataset, prefix=f"{prefix}{i}_")
            df = dataset.to_pandas()
            df.to_csv(f"./distinguisher/tmp/{prefix}tmp.csv")
        df = dataset.to_pandas()
        df[f"{prefix}A_normal"] = df[[f"{prefix}{i}_choice" for i in range(count)]].apply(lambda row: (row == 'A').sum(), axis=1)
        df[f"{prefix}B_normal"] = df[[f"{prefix}{i}_choice" for i in range(count)]].apply(lambda row: (row == 'B').sum(), axis=1)
        df[f"{prefix}A_flipped"] = df[[f"{prefix}{i}_flipped_choice" for i in range(count)]].apply(lambda row: (row == 'A').sum(), axis=1)
        df[f"{prefix}B_flipped"] = df[[f"{prefix}{i}_flipped_choice" for i in range(count)]].apply(lambda row: (row == 'B').sum(), axis=1)

        # drop the individual columns
        df = df.drop(columns=[f"{prefix}{i}_choice" for i in range(count)])
        df = df.drop(columns=[f"{prefix}{i}_flipped_choice" for i in range(count)])
        
        df[f"{prefix}A_count"] = df[f"{prefix}A_normal"] + df[f"{prefix}A_flipped"]
        df[f"{prefix}B_count"] = df[f"{prefix}B_normal"] + df[f"{prefix}B_flipped"]
        df[f"{prefix}choice"] = df.apply(lambda x: "A" if x[f"{prefix}A_count"] > x[f"{prefix}B_count"] else "B", axis=1)
        df[f"{prefix}flipped_choice"] = df.apply(lambda x: "A" if x[f"{prefix}A_count"] >= x[f"{prefix}B_count"] else "B", axis=1)

        # normalize the counts
        df[f"{prefix}A_normal"] /= count
        df[f"{prefix}B_normal"] /= count
        df[f"{prefix}A_flipped"] /= count
        df[f"{prefix}B_flipped"] /= count
        df[f"{prefix}A_count"] /= count*2
        df[f"{prefix}B_count"] /= count*2

        return datasets.Dataset.from_pandas(df)

    @property
    @abstractmethod
    def input_keys(self):
        """
        Do not include "A" or "B" as input; they are added automatically
        """
        pass

    @property
    @abstractmethod
    def output_keys(self):
        """
        Important: must include "choice" property that returns either "A" or "B", representing the distinguisher's answer
        """
        pass