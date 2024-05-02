import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

class RCRGMetric:
    def __init__(self, n_bins=15, sigma=0.05):
        self.n_bins = n_bins
        self.sigma = sigma

    def calculate_rcr_metric(self, output, target):
        output = torch.softmax(output, dim=1)
        batch, classes = output.shape
        total_rcr_metric = 0.0

        for c in range(classes):
            conf_gaussian = torch.zeros(batch).cuda()
            avg_count = (target == c).float().mean()

            for i in range(batch):
                gaussian = torch.distributions.normal.Normal(output[i, c], torch.tensor([self.sigma]).cuda())
                g = gaussian.log_prob(output[:, c]).exp()
                conf_gaussian += output[i, c] * (g / torch.sum(g))
            conf_gaussian /= batch

            avg_conf = torch.mean(conf_gaussian)
            rcr_metric = torch.abs(avg_conf - avg_count)
            total_rcr_metric += rcr_metric.item()

        return total_rcr_metric / classes  # Average over classes

class RCRMMetric:
    def __init__(self, n_bins=15, sigma=0.05):
        self.n_bins = n_bins
        self.sigma = sigma

    def calculate_rcr_metric(self, model,data, target, num_augmentations=10, num_components=3):
        assert len(data.shape) == 4, "Data should be in the format (batch_size, channels, height, width)"
        assert data.shape[0] == target.shape[0], "Batch sizes of data and target don't match"
        assert len(target.shape) == 1, "Invalid shape for target"
        
        device = data.device
        batch_size = data.shape[0]
        total_rcr_metric = 0.0

        for i in range(batch_size):
            data_aug = data[i].unsqueeze(0).repeat(num_augmentations, 1, 1, 1)  # Apply test-time augmentation
            output = model(data_aug)  # Get model predictions

            # Reshape and softmax
            output = output.view(num_augmentations, -1)
            output = F.softmax(output, dim=1)

            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=num_components)
            gmm.fit(output.cpu().detach().numpy())

            # Estimated parameters of the mixture model
            means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
            weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=device)
            stds = torch.tensor(gmm.covariances_, dtype=torch.float32, device=device).sqrt()

            # Calculate RCR metric
            avg_conf = (means * weights.view(-1, 1)).sum(dim=0).mean()
            avg_count = (target[i] == torch.arange(output.shape[1], device=device)).float().mean()
            rcr_metric = torch.abs(avg_conf - avg_count)
            #avg_conf = (means * weights).sum(dim=1).mean()
            #avg_count = (target[i] == torch.arange(output.shape[1], device=device)).float().mean()
            #rcr_metric = torch.abs(avg_conf - avg_count)
            total_rcr_metric += rcr_metric.item()

        return total_rcr_metric / batch_size  # Average over samples

# Example of usage:
#model = net() #stand in for machine learning model
#RCR = RCRMetric()

# Random Data Sample
#data = torch.randn(10, 37).cuda()  # Example data
#targets = torch.randint(0, 37, (10,)).cuda()  # Example targets

# Model Outputs
#output = model(data)

# Calculate RCR metric
#rcr_metric = RCR.calculate_rcr_metric(output, targets)
#print(f"RCR Metric: {rcr_metric}")