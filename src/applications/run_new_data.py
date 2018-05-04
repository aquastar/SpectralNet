# run spectral net
from src.applications.spectralnet import run_net
import _pickle as pk

data = pk.load(open('../data.pk','rb'))
x_spectralnet, y_spectralnet = run_net(data, params)