"""
Implementation of reporter classes, which are triggered on particular events. Reporters
are generally intended to  provide information to the user, store checkpoints, etc.
"""

import time

from neat.math_util import mean, stdev

from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import math
import statistics
import os

class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """

    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""

    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            print("   ID   age  size   fitness   adj fit  stag")
            print("  ====  ===  ====  =========  =======  ====")
            for sid in sorted(species_set.species):
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                st = self.generation - s.last_improved
                print(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)
class TBReporter(BaseReporter):
    """Uses `print` and TensorBoard to output information about the run;"""
    def __init__(self, print_species_detail,gen_buff, runs,datte):
        self.print_species_detail = print_species_detail
        self.gen_buff = gen_buff
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0                                                            #delete species
        #self.gen_data=[]
        self.spa = 15
        ##TensorBoard init
        LOG_DIR = "logs/neat-gym-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        #self.logger = SummaryWriter(f"logs/{datte}/{runs:03d}", comment="",flush_secs=180)
        self.logger = SummaryWriter(f"logs/{datte}/{runs}", comment="", flush_secs=180)

        #todo hyper parmetere do tensorboard:
        # self.logger.add_hparams(
        #     {'lr': 0.1 * i, 'bsize': i},
        #     {'hparam/accuracy': 10 * i,
        #      'hparam/loss': 10 * i})
        
    # print actual gen.    
    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()
        
    def end_generation(self, config, population, species_set):
        #todo Max/mean Fitness all pop
        pop_len     = len(population)
        specie_len  = len(species_set.species)
        if specie_len > self.spa:
            self.spa =  specie_len
        self.logger.add_scalar(f"PopCount", pop_len, self.generation)
        self.logger.add_scalar(f"SpeciesCount", specie_len, self.generation)
        if self.print_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(pop_len, specie_len))
            print("   ID   age  size   fitness   adj fit  stag")
            print("  ====  ===  ====  =========  =======  ====")

        self.specie_data2array(species_set.species)

        # old code
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        self.logger.add_scalar(f"Best specie", best_species_id, self.generation)
        self.logger.add_scalar(f"Best Genome fitness", best_genome.fitness, self.generation)
        self.logger.add_scalar(f"Size of Best/nodes", best_genome.size()[0], self.generation)
        self.logger.add_scalar(f"Size of Best/connections", best_genome.size()[1], self.generation)
        self.logger.add_scalar(f"Mean Pop fitness", fit_mean, self.generation)
        self.logger.add_scalar(f"Stdev Pop fitness", fit_std, self.generation)

        #self.logger.text    #todo add text best etc t

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))
        self.logger.add_custom_scalars(self.Layout_species())
        with open('config', 'r') as f:
            self.logger.add_text("conf",f.read())
        self.logger.close()


    def species_stagnant(self, sid, species):
        if self.print_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        pass
    def specie_data2array(self,species):
        #specieArr=[] #TODO buffer for n generations
        maxSFit=-100000
        for sid in sorted(species):                                                                         # iter Species
            specie      = species[sid]                                                                      # specie
            age         = self.generation - specie.created                                                  # age of specie
            n           = len(specie.members)                                                               # member of specie   
            fi          = 0 if specie.fitness is None else specie.fitness                                   #fit
            af          = 0 if specie.adjusted_fitness is None else specie.adjusted_fitness
            stag        = self.generation - specie.last_improved
            
            self.logger.add_scalar(f"MemberCount/sid_{sid}", n, self.generation)
            maxFit = specie.get_MaxFitness()
            if maxFit is not None:
                self.logger.add_scalar(f"MaxFitness/sid_{sid}", maxFit,self.generation) #todo bude lepsie spravit to , ze ak to bude None tak to nezapise do tensorboardu
                if (maxFit > maxSFit):
                    # print(f"-------------------------->max Fitness {maxFit} specie: {sid}")
                    maxSFit = maxFit
            MeanFit = specie.get_MeanFitness()
            if MeanFit is not None:
                self.logger.add_scalar(f"MeanFitness/sid_{sid}", MeanFit, self.generation)
            #print("ALL ALL", specie.get_fitnesses())
            #print("MAX MAX",specie.get_MaxFitness())
            #print(f"Spiece:{specie.fitness:.3f}")
            #self.logger.add_scalar(f"MeanFitness/sid_{sid}", mean(fi), self.generation)
            if self.print_species_detail:
                print(f"  {sid:>4}  {age:>3}  {n:>4}  {fi:>9}  {af:>7}  {stag:>4}")

            # specieArr.append([sid, age, n, f, af, stag ])
            # return specieArr

    ### create layout for species
    def Layout_species(self):
        layout = {
        'Species':
            {
                'MemberCount per Specie':['Multiline',[f"MemberCount/sid_{n}" for n in range(self.spa)]],
                'Fitness per Specie':['Multiline',[f"MaxFitness/sid_{n}" for n in range(self.spa)]]
            }# Member , Max fitness, mean fitness, ... pre species
            }
        return layout
    ### Console print
    def get_bestGenome(self):
        pass


