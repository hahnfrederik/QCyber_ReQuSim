import sys

sys.path.insert(0, '../src/requsim/quantum_objects')
sys.path.insert(1, '../src/requsim/libs')
from requsim.world import World
from requsim.quantum_objects import Station
from source import SourceMult
import matrix as mat


world = World()

station_a = Station(world=world, position=np.array([0, 2000]))
station_b = Station(world=world, position=np.array([-1000, -1000]))
station_c = Station(world=world, position=np.array([-1000, -1000]))
source = SourceMult(world = world, position = np.array([0,0]), target_stations=[station_a, station_b, station_c])


