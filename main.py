import os
from uuid import uuid4
from scipy.optimize import minimize
from copy import deepcopy
from math import ceil, floor
from itertools import product, chain
import numpy as np
from datetime import datetime
from pyjsoncanvas import (
    Canvas,
    TextNode,
    Edge,
    Color
)
import json


def generate_production_tree(resource, amount, allowed_methods, allowed_resources, recipes, tree=None, dependency=None):
    """
    Needed to generate basic structure of crafting tree. Up until now worked fine. There are some redundant things, and
    misused variables, as it worked previously in the other way.
    :param resource:
    :param amount:
    :param allowed_methods:
    :param allowed_resources:
    :param recipes:
    :param tree:
    :param dependency:
    :return:
    """
    if tree is None:
        tree = {"resource": resource, "amount": amount, "methods": []}

    if dependency is None:
        dependency = []

    if dependency.count(resource) > 1:
        tree["loop"] = True
        return tree

    dependency.append(resource)

    for recipe_name, recipe_data in recipes.items():

        if any(output["name"] == resource for output in recipe_data["outputs"]) and recipe_name in allowed_methods:

            for input_combination in recipe_data.get("input_combinations", [recipe_data.get("inputs", [])]):

                method_tree = {
                    "method": recipe_name,
                    "dependency": dependency,
                    "recipe": {"input": input_combination, "output": [i for i in recipe_data["outputs"] if i["name"] == resource][0]},
                    "input_speed": {},
                    "sub_trees": []
                }

                # Calculate the required input resources for the current method and input combination
                for input_data in input_combination:
                    input_resource = input_data["name"]
                    input_amount = input_data["amount"]

                    if input_resource in allowed_resources:
                        sub_tree = generate_production_tree(
                            input_resource,
                            input_amount,
                            allowed_methods,
                            allowed_resources,
                            recipes,
                            tree=None,
                            dependency=dependency.copy()
                        )

                        method_tree["sub_trees"].append(sub_tree)

                        method_tree["input_speed"][input_resource] = None

                tree["methods"].append(method_tree)

    dependency.pop()

    return tree


def get_time(name):
    """
    Return time to process recipe
    :param name:
    :return:
    """
    return recipes[name]["time"]


class Machine:
    """
    Main node of simulation line of which used to calculate total time of until needed amount of needed resource is
    generated, to allow  in the future creation of the best amount of instances for each machine.
    """
    def __init__(self, name, resources1, resources2, recipe, time, links, uid):
        self.name = name
        self.resources_i = resources1
        self.recipe = recipe
        self.resources_o = resources2
        self.time = time
        self.give_links = links
        self.counter = 0
        self.uid = uid
        self.instances = 1

    def run(self):

        test = [
            self.resources_i[i_comb["name"]] >= i_comb["amount"] * self.instances
            for i_comb in self.recipe["input"]
        ]
        if all(test):
            if self.time != 0: #  If recipe take time to process
                self.counter += 1
                if self.counter == self.time:
                    self.counter = 0
                    if len(self.resources_i) > 0:
                        for i_comb in self.recipe["input"]:
                            self.resources_i[i_comb["name"]] -= i_comb["amount"] * self.instances
                    self.resources_o[self.recipe["output"]["name"]] += (self.recipe["output"]["amount"] *
                                                                        float(1
                                                                              if self.recipe["output"][
                                                                                     "probability"] == 1
                                                                              else self.recipe["output"][
                                                                                       "probability"] *
                                                                                   0.8) * self.instances)
            else: #  If take 0 time (like crafting table, so manual things, or difficult to calculate time to, like
                #  in-world-actions recipes; down there is if it has inputs then craft all possible, if no make it
                #  infinite (maybe that is absurd, but just in case)
                if len(self.resources_i) > 0:
                    amount_to_craft = 1
                    used_resources = {index: i_comb["amount"] * self.instances for index, i_comb in
                                      enumerate(self.recipe["input"])}
                    resources_dictionary = {index: i_comb["name"] for index, i_comb in enumerate(self.recipe["input"])}
                    resources_dictionary.update({v: k for k, v in resources_dictionary.items()})

                    amount_to_craft = min([floor(i / (self.recipe["input"][resources_dictionary[key]]["amount"] *
                                                  self.instances)) for key, i in self.resources_i.items()])

                    for resource1, value in used_resources.items():
                        self.resources_i[resources_dictionary[resource1]] -= value
                    self.resources_o[self.recipe["output"]["name"]] += (
                                amount_to_craft * self.recipe["output"]["amount"] * float(1
                                                                                          if self.recipe["output"][
                                                                                                 "probability"] == 1
                                                                                          else self.recipe["output"][
                                                                                                   "probability"] *
                                                                                               0.8) * self.instances)
                else:
                    self.resources_o[self.recipe["output"]["name"]] = float("inf")


    def transport(self, line):
        """
        After each set of operation within the same step it move all output resources further by chain
        :param line:
        :return:
        """
        for m_uid in self.give_links:
            machine = next((i for i in line if i.uid == m_uid), None)
            max_resource_amount = min(max(
                max_instance_speed[t] for t in self.recipe["output"]["type"]
            ), self.resources_o[self.recipe["output"]["name"]])
            machine.resources_i[self.recipe["output"]["name"]] += max_resource_amount
            self.resources_o[self.recipe["output"]["name"]] -= max_resource_amount


def generate_tree(tree, parent_id):
    """Processing of original generated crafting tree to generate list of all possible lines of processing"""
    data = list(chain(*[generate_methods(m, parent_id, tree) for m in tree["methods"]]))
    return data


def generate_methods(method, parent_id, tree):
    """Processing of original generated crafting tree to generate list of all possible lines of processing"""
    method_id = uuid4()
    method_item = Machine(method["method"],
                          {key: 0 for key in list(method["input_speed"].keys())},
                          {tree["resource"]: 0},
                          method["recipe"],
                          get_time(method["method"]),
                          [parent_id] if parent_id != [] else [],
                          method_id)
    data = [generate_tree(s, method_id) for s in method["sub_trees"]]
    if all(len(i) != 0 for i in data) and data:
        data = list(product(*data + [[method_item]]))
        if all(type(i[0]) == tuple for i in data):
            data = [elem[0] + (elem[1],) for elem in data]

    else:
        data = [method_item]
    return data


def optimize_instances(tree, target_time, amount, name):
    machines = generate_tree(tree, [])
    print(f"{len(machines)} machine lines has generated")
    results = []
    for index, line in enumerate(machines):
        optimal_Ms = find_optimal_Ms(line, target_time, amount, name)
        print(f"base optimal Ms for {index + 1} / {len(machines)} line found")
        #  Optimizing resulted amount of instances as COBYLA from scipy.optimize.minimize can give floats
        optimal_Ms = [(floor(i), ceil(i)) for i in optimal_Ms]
        optimal_Ms_product = list(set(product(*optimal_Ms)))
        total_times = [calculate_total_time(variant, line, amount, name) for variant in optimal_Ms_product]
        differences = [abs(target_time - time) for time in total_times]
        best_difference = min(differences)
        best_Ms = [
            (optimal_Ms_product[i], total_times[i])
            for i in list(range(len(differences)))
            if differences[i] == best_difference
        ]
        results.extend({"line": line, "M": M, "t": T} for M, T in best_Ms)
    print("best lines chosen")
    return sorted(results, reverse=True, key=lambda x: abs(x["t"] - target_time))

def find_optimal_Ms(line, target_time, amount, name):
    """Optimizing to get best amount of instances; can be floats"""
    Ms_initial = [1 for _ in range(len(line))]
    bounds = [(1, None) for _ in range(len(line))]

    result = minimize(objective, np.array(Ms_initial), args=(line, target_time, amount, name), bounds=bounds, method='COBYLA')
    return result.x


def objective(mults, line, target_time, amount, name):
    """Objective for minimize"""
    total_time = calculate_total_time(mults, line, amount, name)
    return (total_time - target_time) ** 2


def calculate_total_time(mults, line, amount, name):
    """Main node of simulation, calculate total steps needed to get to the requirements from given line, and amount of
    instances for each machine."""
    line = deepcopy(line)
    for i, m in zip(line, mults):
        i.instances = m
        for_inf = [
            r
            for r in list(i.resources_i.keys())
            if r
            not in [
                s.recipe["output"]["name"]
                for s in [g for g in line if g.give_links == [i.uid]]
            ]
        ]
        for key in for_inf:
            i.resources_i[key] = float("inf") #  Give infinite for deepest resources that have no resource to create
                                              #  them from.

    steps = 0
    while line[-1].resources_o[name] < amount:
        steps += 1
        for machine in line:
            machine.run()
        for machine in line:
            machine.transport(line)
    return steps


def generate_full_tree(resource, amount, target_time, allowed_methods, allowed_resources, recipes):
    """Main function"""
    production_tree = generate_production_tree(resource, amount, allowed_methods, allowed_resources, recipes)
    print("base production tree has generated")
    best_lines_dict = optimize_instances(production_tree, target_time, amount, resource)
    print("best lines found")
    t = str(datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
    os.mkdir(f"Results/{t}")
    c = 0
    # Draw in Obsidian canvas json
    for line_variant in best_lines_dict[:5]:
        canvas = Canvas(nodes=[], edges=[])
        main_node = TextNode(x=0, y=-500, width=400, height=150, text=f"# **Task: Produce {amount} of {resource} in {target_time} t**")
        canvas.add_node(main_node)
        line = line_variant["line"]
        Ms = list(line_variant["M"])

        canvas_nodes, canvas_edges = create_tree_layout(line, Ms, [(resource, amount)])
        for node in canvas_nodes:
            canvas.add_node(node)
        for edge in canvas_edges:
            canvas.add_edge(edge)
        json_str = canvas.to_json()
        with open(f"Results/{t}/Ms-{', '.join(list(map(str, line_variant['M'])))}; T-{str(line_variant['t'])};Id-{c}.canvas", "w") as f:
            f.write(json_str)
        c += 1
    print("data stored")




class MyTextNode(TextNode):
    parent_uid: uuid4

    def __init__(self, parent_uid: uuid4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_uid = parent_uid

def create_tree_layout(nodes, multipliers, requirements, canvas_nodes=None, canvas_edges=None, parent_id=None, level=0, x_offset=0, y_offset=0, x_spacing=1250, y_spacing=1800):
    """Create canvas for one line"""
    if parent_id is None:
        parent_id = []
    if canvas_nodes is None:
        canvas_nodes = []
    if canvas_edges is None:
        canvas_edges = []

    for node, M in zip(nodes, multipliers):
        if node.give_links == parent_id:
            # Calculate the position and size of the node
            x = x_offset + len(canvas_nodes) * x_spacing
            y = y_offset + level * y_spacing
            width = 1000
            height = 1500

            needed_amount = next((i[1] for i in requirements if i[0] == node.recipe["output"]["name"]), None)
            generated_amount = node.recipe["output"]["amount"] * float(1 if node.recipe["output"]["probability"] == 1 else node.recipe["output"]["probability"] * 0.8)
            for inp in node.recipe["input"]:
                requirements.append((inp["name"], needed_amount / generated_amount * inp["amount"]))

            # Create a TextNode with the calculated position and size
            canvas_node = MyTextNode(node.uid, x=x, y=y, width=width, height=height,
                                     text=generate_data_for_node(M, node, requirements))
            canvas_nodes.append(canvas_node)

            # Create an Edge between the parent and the current node
            if parent_id:
                if parent_node := next(
                    (n for n in canvas_nodes if [n.parent_uid] == parent_id),
                    None,
                ):
                    edge = Edge(
                        fromNode=parent_node.id,
                        fromSide="bottom",
                        toNode=canvas_node.id,
                        toSide="top",
                        color=Color("#000000"),
                    )
                    canvas_edges.append(edge)

            # Recursively create the child nodes
            create_tree_layout(
                nodes, multipliers, requirements, canvas_nodes, canvas_edges, [node.uid], level + 1, x, y, x_spacing, y_spacing
            )

    return canvas_nodes, canvas_edges


def generate_data_for_node(M, node, recqs):
    """Create test for node with needed data"""
    output_speed = f'{node.recipe["output"]["amount"] * float(1 if node.recipe["output"]["probability"] == 1 else node.recipe["output"]["probability"] * 0.8) * M / recipes[node.name]["time"]} / t' if recipes[node.name]["time"] != 0 else "It is no time operation"
    text = f"""# **{node.name} X {M}**
![[{recipes[node.name]["foto"]}]]
## Recipe used:
### Output:
- Resource: {node.recipe["output"]["name"]}; Amount: {node.recipe["output"]["amount"] * 
 float(1 if node.recipe["output"]["probability"] == 1 else node.recipe["output"]["probability"] * 0.8)} * {M} = {output_speed}; Total amount: {next((rt[1] for rt in recqs if rt[0] == node.recipe["output"]["name"]), None)}"""
    recqs.pop([recqs.index(i) for i in recqs if i[0] == node.recipe["output"]["name"]][0])
    rec = "\n".join([f'- Resource: {i["name"]}; Speed: {i["amount"]}  * {M}  = {str((i["amount"] * M) / recipes[node.name]["time"]) + " / t" if recipes[node.name]["time"] != 0 else "It is no time operation"}; Total amount: {next((rt[1] for rt in recqs if rt[0] == i["name"]), None)}' for i in node.recipe["input"]])

    text1 = f"""### Input:
{rec}"""
    return text + "\n" + text1


# Example usage
if __name__ == "__main__":
    with open("database.json", "r") as f:
        data = json.load(f)
    recipes = data["recipes"]
    resources = data["resources"] #  Needed to restrict resources, can be implemented in some other way


    not_allowed_methods = []
    not_allowed_resources = []

    resource = "minecraft:cobblestone"
    amount = 100
    needed_time = 10

    not_allowed_recipes = [key for key, value in recipes.items() if value["method"] in not_allowed_methods]
    recipes_names = list(recipes)

    allowed_methods = [i for i in recipes_names if i not in not_allowed_recipes]
    allowed_resources = [resource] + [
        i for i in resources if i not in not_allowed_resources
    ]
    #  Speed restriction in element per choosen time element
    max_instance_speed = {"gas": 250, "liquid": 250, "energy": 1000, "item": 25}

    generate_full_tree(resource, amount, needed_time, allowed_methods, allowed_resources, recipes)

