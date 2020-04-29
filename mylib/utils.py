from typing import List
def nice_string_output(
    names: List[str], values: List[str], extra_spacing: int = 0,
):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(
            name,
            value,
            spacing=extra_spacing + max_values + max_names - len(name),
        )
    return string[:-2]
