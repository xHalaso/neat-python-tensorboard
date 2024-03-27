import neat
from neat.genome import DefaultGenome
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from itertools import count

class FixedFloatAttribute(BaseAttribute):
    """
    Class for floating-point numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]}
    mutate_value = 0.1

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError(f"Unknown init_type {getattr(config, self.init_type_name)!r} for {self.init_type_name!s}")

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + random.uniform(-mutate_value, mutate_value), config)

        replace_rate = getattr(config, self.replace_rate_name)

        r = random()
        if r < replace_rate:
            return self.init_value(config)

        return value

    def validate(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")

class FixedStructureNodeGene(BaseGene):
    _gene_attributes = [
                        FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options=''),
                        StringAttribute('aggregation', options='')]

    def distance(self, other, config):
        return 0.0

class FixedStructureConnectionGene(BaseGene):
    _gene_attributes = [FixedFloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def mutate(self, config):
        # Mutate only weights
        a = self._gene_attributes[0]
        v = getattr(self, a.name)
        setattr(self, a.name, a.mutate_value(v, config))

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        return d
    
class FixedStructureGenomeConfig(object):
    __params = [ConfigParameter('feed_forward', bool),
                ConfigParameter('num_inputs', int),
                ConfigParameter('num_outputs', int),
                ConfigParameter('num_hidden', int),
                ConfigParameter('num_hidden_layers', int),
                ConfigParameter('initial_connection', str, 'full_direct'),
                ConfigParameter('conn_add_prob', float),
                ConfigParameter('conn_delete_prob', float),
                ConfigParameter('node_add_prob', float),
                ConfigParameter('node_delete_prob', float)]


    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        self.activation_options = params.get('activation_options', 'sigmoid').strip().split()
        self.aggregation_options = params.get('aggregation_options', 'sum').strip().split()
        self.aggregation_function_defs = AggregationFunctionSet()

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self.__params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self.__params += self.connection_gene_type.get_config_params()
        self.connection_fraction = None
        self.node_indexer = None
        self.structural_mutation_surer = 'false'

        # Use the configuration data to interpret the supplied parameters.
        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            if node_dict:
                self.node_indexer = count(max(list(node_dict)) + 1)
            else:
                self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id
    
    def save(self, f):
        write_pretty_params(f, self, self.__params)

class FixedStructureGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = FixedStructureNodeGene
        param_dict['connection_gene_type'] = FixedStructureConnectionGene
        return FixedStructureGenomeConfig(param_dict)
    
    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def mutate(self, config):
        """ Mutates this genome. """
        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)
            self.connections[key].enabled = True

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    # def add_connection(self, config, input_key, output_key, weight):
    #     key = (input_key, output_key)
    #     connection = FixedStructureConnectionGene(key)
    #     connection.init_attributes(config)
    #     connection.weight = weight
    #     connection.enabled = True

    #     self.connections[key] = connection

    def distance(self, other, config):
        return 0.0
    
    def size(self):
        """Returns genome 'complexity', taken to be (number of nodes, number of enabled connections)"""
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        for k, ng in self.nodes.items():
            s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s
    
    # def add_hidden_nodes(self, config):
    #     for i in range(config.num_hidden):
    #         node_key = self.get_new_node_key()
    #         assert node_key not in self.nodes
    #         node = self.__class__.create_node(config, node_key)
    #         self.nodes[node_key] = node

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))
        return connections   

    def configure_new(self, config):
        # Create node genes for the input and output nodes
        for node_key in config.input_keys + config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Initialize list to keep track of nodes in the last layer (start with input layer)
        last_layer_nodes = config.input_keys

        # Add hidden nodes and layers
        # config.num_hidden now represents the number of neurons PER HIDDEN LAYER
        if config.num_hidden > 0 and config.num_hidden_layers > 0:
            for _ in range(config.num_hidden_layers):
                layer_node_ids = []
                for _ in range(config.num_hidden):  # Create the specified number of nodes for this layer
                    node_key = config.get_new_node_key(self.nodes)
                    self.nodes[node_key] = self.create_node(config, node_key)
                    layer_node_ids.append(node_key)
                
                # Connect all nodes from the last layer to the current layer's nodes
                for last_layer_node in last_layer_nodes:
                    for current_layer_node in layer_node_ids:
                        connection = self.create_connection(config, last_layer_node, current_layer_node)
                        self.connections[connection.key] = connection

                # Update last_layer_nodes to reflect the current layer for next iteration
                last_layer_nodes = layer_node_ids

            # Connect all nodes from the last hidden layer to the output nodes
            for hidden_node_id in last_layer_nodes:
                for output_node_id in config.output_keys:
                    connection = self.create_connection(config, hidden_node_id, output_node_id)
                    self.connections[connection.key] = connection
        else:
            # If there are no hidden layers, connect input nodes directly to output nodes
            for input_node_id in config.input_keys:
                for output_node_id in config.output_keys:
                    connection = self.create_connection(config, input_node_id, output_node_id)
                    self.connections[connection.key] = connection

    @staticmethod
    def create_node(config, node_id):
        node = FixedStructureNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection