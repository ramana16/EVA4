import pprint

class ModelConfig(object):

	def __init__(self,):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 128
		self.batch_size_cpu = 128	
		self.num_workers = 4
		# Regularization
		self.dropout = 0.15
		self.l1_decay = 3e-6
		self.l2_decay = 1e-3
		self.lr = 0.001
		self.momentum = 0.9
		self.epochs = 5

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)

def test_config():
	args = ModelConfig()
	args.print_config()

if __name__ == '__main__':
	test_config()