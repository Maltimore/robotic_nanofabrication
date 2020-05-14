import torch

from . import basemodel


class Model(basemodel.BaseModel):
    def __init__(self, params, n_outputs):
        """__init__
        :param params: dict
            parameters to initialize model
        :param n_outputs: int
            number of outputs
        """
        super(Model, self).__init__()

        self.n_hidden = params['n_hidden']  # number of hidden neurons in each layer
        self.n_outputs = n_outputs
        self.n_features = params['n_features']

        self._initialize()

    def _initialize(self):

        self.dense1 =  torch.nn.Linear(self.n_features, self.n_hidden)
        self.denseHV = torch.nn.Linear(self.n_hidden, self.n_hidden // 2)
        self.denseHA = torch.nn.Linear(self.n_hidden, self.n_hidden // 2)
        self.denseV =  torch.nn.Linear(self.n_hidden // 2, 1)
        self.denseA =  torch.nn.Linear(self.n_hidden // 2, self.n_outputs)

    def forward(self, x):
        x = x.contiguous()
        h = torch.tanh(self.dense1(x))

        hv = torch.tanh(self.denseHV(h))
        v = self.denseV(hv)

        ha = torch.tanh(self.denseHA(h))
        a = self.denseA(ha)
        q = v + a - torch.mean(a, keepdim=True, dim=-1)
        # Q: batch x n_outputs
        # V: batch x 1
        # A: batch x n_outputs
        # Q(state, action) = V(state) + A(state, action) [- 1/n_action sum_action A(state, action)]
        return q, v, a
