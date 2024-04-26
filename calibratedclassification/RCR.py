import torch

class RCRMetric:
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
