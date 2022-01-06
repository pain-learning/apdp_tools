// Norbury et al 2018 eLife, generalisation model with 2 gs and bias

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int cue[N, T];
  int choice[N, T];
  real outcome[N, T];
}

transformed data {
  // initial values
  vector[7] initV;
  // cue probability
  vector[7] pCue;
  // cue rhos
  vector[7] cueR;
  // initial learning rate
  real alpha0;

  initV = rep_vector(0.0, 7);
  pCue = [0.052, 0.264, 0.052, 0.264, 0.052, 0.264, 0.052]';
  cueR = [0.0, 0.25, 0.5, 0.25, 0.5, 0.75, 1.0]';
  alpha0 = 0.7;
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] sigma_a_pr;
  vector[N] sigma_n_pr;
  vector[N] eta_pr;
  vector[N] kappa_pr;
  vector[N] beta_pr;
  vector[N] bias_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] sigma_a;
  vector<lower=0, upper=1>[N] sigma_n;
  vector<lower=0, upper=1>[N] eta;
  vector<lower=0, upper=1>[N] kappa;
  vector<lower=0, upper=10>[N] beta;
  vector<lower=0, upper=1>[N] bias;

  for (i in 1:N) {
    sigma_a[i] = Phi_approx(mu_pr[1] + sigma[1] * sigma_a_pr[i]);
    sigma_n[i] = Phi_approx(mu_pr[2] + sigma[2] * sigma_n_pr[i]);
    eta[i]     = Phi_approx(mu_pr[3] + sigma[3] * eta_pr[i]);
    kappa[i]   = Phi_approx(mu_pr[4] + sigma[4] * kappa_pr[i]);
    beta[i]    = Phi_approx(mu_pr[5] + sigma[5] * beta_pr[i]) * 10;
    bias[i]    = Phi_approx(mu_pr[6] + sigma[6] * bias_pr[i]);
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  sigma_a_pr  ~ normal(0, 1.0);
  sigma_n_pr  ~ normal(0, 1.0);
  eta_pr      ~ normal(0, 1.0);
  kappa_pr    ~ normal(0, 1.0);
  beta_pr     ~ normal(0, 1.0);
  bias_pr     ~ normal(0, 1.0);

  for (i in 1:N) {
    // Define values
    vector[7] Q;
    vector[7] cue_rho; // current cue rho
    vector[7] diff2; // current cue rho
    vector[7] G; // current cue rho

    real pred_V; // previous Q
    real PE; // prediction error
    real alpha; // learning rate
    real sig; // sigma
    real weighted_PE; // G

    int s_cue; // cue of trial

    // Initialize values
    Q    = initV;
    alpha = alpha0;

    for (t in 1:Tsubj[i]) {

      // cue shown drawn from pCue
      // s_cue[i, t] ~ categorical_logit(pCue);
      s_cue = cue[i, t];

      // softmax choice
      if (s_cue==1) {
        pred_V = 0.75*Q[s_cue] + 0.25*Q[2];
      } else if (s_cue==3)  {
        pred_V = 0.75*Q[s_cue] + 0.25*Q[2];
      } else if (s_cue==5)  {
        pred_V = 0.75*Q[s_cue] + 0.25*Q[6];
      } else if (s_cue==7)  {
        pred_V = 0.75*Q[s_cue] + 0.25*Q[6];
      } else {
        pred_V = Q[s_cue];
      }
      // avoid probability
      choice[i, t] ~ bernoulli(inv_logit(beta[i] * (0.0 - pred_V - 0.2 - bias[i])));

      // update values (avoided or not)
      if (choice[i, t] != 0) {
        PE = 0.0;
        Q = Q;
      } else {
        // Prediction error signals
        PE     = outcome[i, t] - Q[s_cue];
        // define sigma
        if (outcome[i, t] == 0.0) {
            sig = sigma_n[i];
        } else {
            sig = sigma_a[i];
        }
        // update values
        cue_rho = rep_vector(cueR[s_cue],7);
        diff2 = (cueR-cue_rho).*(cueR-cue_rho);
        diff2 = diff2 ./ (2*sig^2);
        G = 1.0 ./ exp(diff2);
        weighted_PE = kappa[i] * alpha * PE;
        Q += weighted_PE * G;
      }
      // update alpha
      alpha = eta[i]*fabs(PE) + (1-eta[i])*alpha;
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_sigma_a;
  real<lower=0, upper=1> mu_sigma_n;
  real<lower=0, upper=1> mu_eta;
  real<lower=0, upper=1> mu_kappa;
  real<lower=0, upper=10> mu_beta;
  real<lower=0, upper=1> mu_bias;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_sigma_a = Phi_approx(mu_pr[1]);
  mu_sigma_n = Phi_approx(mu_pr[2]);
  mu_eta     = Phi_approx(mu_pr[3]);
  mu_kappa   = Phi_approx(mu_pr[4]);
  mu_beta    = Phi_approx(mu_pr[5]) * 10;
  mu_bias    = Phi_approx(mu_pr[6]);

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[7] Q;
      vector[7] cue_rho; // current cue rho
      vector[7] diff2; // current cue rho
      vector[7] G; // current cue rho

      real pred_V; // previous Q
      real PE; // prediction error
      real alpha; // learning rate
      real sig; // sigma
      real weighted_PE; // G

      int s_cue; // cue of trial

      // Initialize values
      Q    = initV;
      alpha = alpha0;
      log_lik[i] = 0.0;

      for (t in 1:Tsubj[i]) {
        // cue shown drawn from pCue
        // s_cue[i, t] ~ categorical_logit(pCue);
        s_cue = cue[i, t];

        // softmax choice
        if (s_cue==1) {
          pred_V = 0.75*Q[s_cue] + 0.25*Q[2];
        } else if (s_cue==3)  {
          pred_V = 0.75*Q[s_cue] + 0.25*Q[2];
        } else if (s_cue==5)  {
          pred_V = 0.75*Q[s_cue] + 0.25*Q[6];
        } else if (s_cue==7)  {
          pred_V = 0.75*Q[s_cue] + 0.25*Q[6];
        } else {
          pred_V = Q[s_cue];
        }

        // compute log likelihood of current trial
        log_lik[i] += bernoulli_logit_lpmf(choice[i, t] | inv_logit(beta[i] * (0.0 - pred_V - 0.2 - bias[i])));

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(inv_logit(beta[i] * (0.0 - pred_V - 0.2 - bias[i])));

        // update values (avoided or not)
        if (choice[i, t] != 0) {
          PE = 0.0;
          Q = Q;
        } else {
          // Prediction error signals
          PE     = outcome[i, t] - Q[s_cue];
          // define sigma
          if (outcome[i, t] == 0.0) {
              sig = sigma_n[i];
          } else {
              sig = sigma_a[i];
          }
          // update values
          cue_rho = rep_vector(cueR[s_cue],7);
          diff2 = (cueR-cue_rho).*(cueR-cue_rho);
          diff2 = diff2 ./ (2*sig^2);
          G = 1.0 ./ exp(diff2);
          weighted_PE = kappa[i] * alpha * PE;
          Q += weighted_PE * G;
        }
        // update alpha
        alpha = eta[i]*fabs(PE) + (1-eta[i])*alpha;
      }
    }
  }
}

