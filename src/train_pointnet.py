import torch
import hdf5storage
import numpy as np
import os
#%%
from tqdm import tqdm

rand_seed = 1234
torch.manual_seed(rand_seed)
#%%
params = {'N_units': 30,
          'L_rate_AE': 1e-4,
          'B_size': 64,#16,
          'Epochs': 3000,
          'N_evals': 30,
          'L_rate_N1N2': 2e-4,
          'W_N1': 1e-4,
          'W_N2': 1e-4,
          'enc_layers': [300, 200],
          'dec_layers': [200],
          'AE_activation': 'tanh',
          'N1_layers': [80, 160, 320, 640, 320, 160, 80],
          'N2_layers': [80, 160, 320, 640, 320, 160, 80],
          'datastep': 10,
          'bat_n_all': True,
          'actual_epoch': 0,
          }
#%%
PI = hdf5storage.loadmat('../data/PI_coma.mat')  # Point clouds Indexes
PI = np.squeeze(PI['PI'])-1;
#%%
data = hdf5storage.loadmat('../data/coma_FEM.mat') # Load dataset
#%%
pix1 = data['meshes_noeye'].astype('float32') # Vertices of the meshes
pix = pix1[:,PI]
outliers = np.asarray([6710, 6792, 6980])-1
remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000])-1
save_every = 100
test_every = 20
#%%
def distance_matrix(array1, array2):
    """
    arguments:
        array1: the array, size: (batch_size, num_point, num_feature)
        array2: the samples, size: (batch_size, num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (batch_size, num_point, num_point)
    """
    batch_size, num_point, num_features = array1.shape
    expanded_array1 = torch.tile(array1, dims=(1, num_point, 1))
    expanded_array2 = torch.reshape(
        torch.tile(torch.unsqueeze(array2, 2),
                   (1, 1, num_point, 1)),
        (batch_size, -1, num_features))

    distances = torch.linalg.norm(expanded_array1-expanded_array2, dim=-1)
    distances = torch.reshape(distances, (batch_size, num_point, num_point))
    return distances


def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (batch_size, num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances, _ = torch.min(distances, dim=-1)
    distances = torch.mean(distances, dim=-1)
    return distances

def av_dist_sum(array1, array2):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2

def chamfer_distance(array1, array2):
    return torch.mean(av_dist_sum(array1, array2))
#%%
model_name = 'final_pnet'
model_path = '../models/' + model_name
if not os.path.exists(model_path):
    os.mkdir(model_path)
#%%
# load datasets

# Eigenvalues of the meshes
e = data['noeye_evals_FEM3'][:,1:params['N_evals']+1].astype('float32')

test_subj = np.arange(18531,20465)-1
idxs_for_train = [int(x) for x in np.arange(0,pix.shape[0],params['datastep']) if (int(x) not in test_subj and int(x) not in outliers and int(x) not in remeshed)]
idxs_for_test = [x for x in np.arange(0,pix.shape[0]) if x not in idxs_for_train and x not in outliers]

train_images = pix[idxs_for_train, :,:]
train_eigs = e[idxs_for_train]

test_images = pix[idxs_for_test, :,:]
test_eigs = e[idxs_for_test]
#%%
class Encoder(torch.nn.Module):

    def __init__(self, n_points):
        super(Encoder, self).__init__()
        self.n_points = n_points
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(1,3), stride=(1,1), padding='valid')
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1,1), padding='valid')
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.max_pool = torch.nn.MaxPool2d((n_points, 1))
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, int(params['N_units']))

    def forward(self, x):
        # input dims (N, P, 3)
        x = x.unsqueeze(1) # (N, 1, P, 3)
        x = self.conv1(x) # (N, 64, P, 1)
        x = self.bn1(x) # (N, 64, P, 1)
        x = self.conv2(x) # (N, 128, P, 1)
        x = self.bn2(x) # (N, 128, P, 1)
        x = self.max_pool(x) # (N, 128, 1, 1)
        x = x.reshape(-1, 128) # (N, 128)
        x = self.linear1(x) # (N, 64)
        x = torch.tanh(x)
        x = self.linear2(x) # (N, N_units)
        x = torch.tanh(x)
        return x

class Decoder(torch.nn.Module):

    def __init__(self, n_points, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_points = n_points
        dec_layers = params['dec_layers'].copy()
        dec_layers.insert(0, self.latent_dim)
        self.layers = torch.nn.ModuleList([])
        for n in range(1, len(dec_layers)):
            self.layers.append(torch.nn.Linear(dec_layers[n - 1], dec_layers[n]))
            self.layers.append(torch.nn.Tanh())
        self.out = torch.nn.Linear(dec_layers[-1], self.n_points * 3)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = x.reshape(-1, self.n_points, 3) # (N, n_points, 3)
        return x

class N1(torch.nn.Module):

    def __init__(self):
        super(N1, self).__init__()
        self.n_evals = params['N_evals']
        self.bn1 = torch.nn.BatchNorm1d(self.n_evals)
        self.layers = torch.nn.ModuleList()
        n1_layers = params['N1_layers'].copy()
        n1_layers.insert(0, self.n_evals)
        for n in range(1, len(n1_layers)):
            self.layers.append(torch.nn.Linear(n1_layers[n - 1], n1_layers[n]))
            self.layers.append(torch.nn.SELU())
            if params['bat_n_all']:
                self.layers.append(torch.nn.BatchNorm1d(n1_layers[n]))
        self.out = torch.nn.Linear(n1_layers[-1], params['N_units'])

    def forward(self, x):
        # input dims: (N, 30)
        x = self.bn1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

class N2(torch.nn.Module):

    def __init__(self):
        super(N2, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(params['N_units'])
        self.layers = torch.nn.ModuleList()
        n2_layers = params['N2_layers'].copy()
        n2_layers.insert(0, params['N_units'])

        for n in range(1, len(n2_layers)):
            self.layers.append(torch.nn.Linear(n2_layers[n - 1], n2_layers[n]))
            self.layers.append(torch.nn.SELU())
            if params['bat_n_all']:
                self.layers.append(torch.nn.BatchNorm1d(n2_layers[n]))
        self.out = torch.nn.Linear(n2_layers[-1], params['N_evals'])

    def forward(self, x):
        # input dims: (N, 30)
        x = self.bn1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

class AE(torch.nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder(n_points=pix.shape[1])
        self.decoder = Decoder(n_points=pix.shape[1], latent_dim=params['N_units'])
        self.n1 = N1()
        self.n2 = N2()

    def forward(self, x, eig):
        lat = self.encoder(x)
        mesh_rec = self.decoder(lat)
        lat_rec = self.n1(eig)
        eval_rec = self.n2(lat)
        return mesh_rec, eval_rec, lat_rec, lat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, lat):
        return self.decoder(lat)

    def predict_eigs(self, x):
        lat = self.encoder(x)
        return self.n1(lat)

    def instant_recovery(self, eig):
        lat_rec = self.n1(eig)
        return self.decoder(lat_rec)
#%%
# Datasets
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_images), torch.tensor(train_eigs))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(params['B_size']), shuffle=True, num_workers=8)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_images), torch.tensor(test_eigs))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(params['B_size']), shuffle=True, num_workers=8)

# Model
model = AE().cuda()

# Optimizer
all_params = set(model.parameters())
wd_params = {model.encoder.linear2.weight}
modules = [model.decoder.layers, model.n1.layers, model.n2.layers]
for m in modules:
    for module in m.modules():
        if isinstance(module, torch.nn.Linear):
            wd_params.add(module.weight)
no_wd = all_params - wd_params

opt = torch.optim.Adam(params=[{"params": list(no_wd)},
                               {"params": list(wd_params), 'weight_decay': 0.01}], lr=params['L_rate_AE'])
#%%
for epoch in range(0, params['Epochs']):

    avg_loss_ae = torch.tensor(0.).cuda()
    avg_loss_N2 = torch.tensor(0.).cuda()
    avg_loss_N1 = torch.tensor(0.).cuda()
    avg_loss = torch.tensor(0.).cuda()

    for meshes, eig in tqdm(train_dataloader):
        g_mesh, g_eval, N1_lat, true_lat = model(meshes.cuda(), eig[:,0:params['N_evals']].cuda())
        loss_ae = chamfer_distance(meshes.cuda(), g_mesh)
        loss_N2 = params['W_N2']*torch.nn.functional.mse_loss(eig[:,0:params['N_evals']].cuda(), g_eval)
        loss_N1 = params['W_N1']*torch.nn.functional.mse_loss(N1_lat, true_lat)
        loss = loss_ae + loss_N2 + loss_N1
        loss.backward()
        opt.step()
        opt.zero_grad()
        avg_loss_ae += loss_ae
        avg_loss_N2 += loss_N2
        avg_loss_N1 += loss_N1
        avg_loss += loss

    avg_loss_ae /= len(train_dataloader)
    avg_loss_N2 /= len(train_dataloader)
    avg_loss_N1 /= len(train_dataloader)
    avg_loss /= len(train_dataloader)

    print(f'Train: avg_loss_ae = {avg_loss_ae.item()}; avg_loss_N1 = {avg_loss_N1.item()}; avg_loss_N2 = {avg_loss_N2.item()}; avg_loss = {avg_loss.item()}')

    if (epoch+1) % save_every == 0:
        torch.save(model, model_path + '/ae_' + str(epoch) + '.pt')

    if (epoch+1) % test_every == 0:
        with torch.no_grad():
            avg_loss_ae = torch.tensor(0.).cuda()
            avg_loss_N2 = torch.tensor(0.).cuda()
            avg_loss_N1 = torch.tensor(0.).cuda()
            avg_loss = torch.tensor(0.).cuda()

            for meshes, eig in tqdm(test_dataloader):
                g_mesh, g_eval, N1_lat, true_lat = model(meshes.cuda(), eig[:,0:params['N_evals']].cuda())
                loss_ae = chamfer_distance(meshes.cuda(), g_mesh)
                loss_N2 = params['W_N2']*torch.nn.functional.mse_loss(eig[:,0:params['N_evals']].cuda(), g_eval)
                loss_N1 = params['W_N1']*torch.nn.functional.mse_loss(N1_lat, true_lat)
                loss = loss_ae + loss_N2 + loss_N1

                avg_loss_ae += loss_ae
                avg_loss_N2 += loss_N2
                avg_loss_N1 += loss_N1
                avg_loss += loss

            avg_loss_ae /= len(test_dataloader)
            avg_loss_N2 /= len(test_dataloader)
            avg_loss_N1 /= len(test_dataloader)
            avg_loss /= len(test_dataloader)
            print(f'Eval: avg_loss_ae = {avg_loss_ae.item()}; avg_loss_N1 = {avg_loss_N1.item()}; avg_loss_N2 = {avg_loss_N2.item()}; avg_loss = {avg_loss.item()}')