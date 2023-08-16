from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import List

import pulp
from pulp import LpVariable as Var, LpConstraint, LpConstraintEQ, LpConstraintLE
from pulp import lpSum, lpDot


@dataclass
class Bid:
    gid: int  # ID of generator that owns this bid
    price: float  # price of bid ($ / MW)
    quantity: float  # quantity being bid (MW)


@dataclass
class Generator:
    gid: int  # generator ID
    name: str  # generator name
    max_ramp_up: float  # max increase in supply between two time steps (MW)
    max_ramp_down: float  # max decrease in supply between two time steps (MW)


@dataclass
class MarketProblem:
    bids: List[List[Bid]]  # for each time step a list of bids
    generators: List[Generator]  # generators (NOTE: guarantee list index = gid)
    demand: List[float]  # demand for each time step (MW)

    @classmethod
    def from_json(cls, data) -> MarketProblem:
        bids = [[Bid(**bid) for bid in bids] for bids in data['bids']]
        generators = [Generator(**gen) for gen in data['generators']]
        return MarketProblem(bids, generators, data['demand'])


@dataclass
class MarketSolution:
    status: str
    objective: float  # ($)
    sup_bid: List[List[float]]  # for each time, supply of each bid (MW)
    sup_gen: List[List[float]]  # for each time, supply of each generator (MW)
    marginal_prices: List[float]  # for each time, marginal price ($ / MW)


def clear_market(prb: MarketProblem) -> MarketSolution:
    m = pulp.LpProblem()

    nt = len(prb.bids)  # number of time steps
    ng = len(prb.generators)  # number of generators

    m += 0.  # TODO

    m.solve()
    return MarketSolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        sup_bid=[[None for _ in bids] for bids in prb.bids],  # TODO
        sup_gen=[[None for _ in range(ng)] for _ in range(nt)],  # TODO
        marginal_prices=[None for _ in range(nt)],  # TODO
    )


def visualise_solution(sol: MarketSolution) -> None:
    """ Your code here for Q2.2 """
    # TODO
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simplex Linear Programming Solver")
    parser.add_argument("file", type=str,
                        help="json model file")
    parser.add_argument("--test", action="store_true",
                        help="test with a smaller test instance")
    args = parser.parse_args()

    data = json.load(open(args.file))
    prb = MarketProblem.from_json(data)
    print(f'Problem data loaded')

    if args.test:
      ng = 3  # number of generators to choose
      prb.generators = prb.generators[:ng]  # first three generators
      demand = 65.0  # change to see what happens
      prb.demand = [demand]  # single time step
      # Only keep bids for first ng generators and first time step
      b_to_keep = []
      for bid in prb.bids[0]:
          if bid.gid < ng:
              b_to_keep.append(bid)
      prb.bids = [b_to_keep]
      for bid in prb.bids[0]: print(bid)

    sol = clear_market(prb)
    print(f'{sol.status} {sol.objective}')
    visualise_solution(sol)
