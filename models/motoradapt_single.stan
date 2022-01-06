// state-space model, single state
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  real rotation[N, T]; // rotations
  real state[N, T]; // observed state
}

transformed data {
  // initial state
  real state0;
  state0 = 0.0;
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] A_pr;
  vector[N] B_pr;
  vector[N] sig_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] A;
  vector<lower=0, upper=1>[N] B;
  vector<lower=0>[N] sig;
  
  for (i in 1:N) {
    A[i] = Phi_approx(mu_pr[1] + sigma[1] * A_pr[i]);
    B[i] = Phi_approx(mu_pr[2] + sigma[2] * B_pr[i]);
    sig[i] = Phi_approx(mu_pr[3] + sigma[3] * sig_pr[i]);
  }
}

model {
  // priors
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  A_pr  ~ normal(0, 1.0);
  B_pr  ~ normal(0, 1.0);
  sig_pr ~ normal(0, 1.0);

  for (i in 1:N) {
    // define values
    real stat; // state

    // initialise value
    stat = state0;

    for (t in 1:Tsubj[i]) {
      // state normally distributed around mean
      state[i, t] ~ normal(stat, sig[i]);
      // update, retained state - learning rate * error
      stat = A[i]*stat - B[i]*(stat-rotation[i, t]);
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_A;
  real<lower=0, upper=1> mu_B;
  real<lower=0> mu_sig;

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

  mu_A = Phi_approx(mu_pr[1]);
  mu_B = Phi_approx(mu_pr[2]);
  mu_sig = Phi_approx(mu_pr[3]);

  { // local section, this saves time and space
    for (i in 1:N) {
      // define values
      real stat; // state

      // initialise value
      stat = state0;

      log_lik[i] = 0;
      
      for (t in 1:Tsubj[i]) {
        // compute log likelihood of current trial
        log_lik[i] += normal_lpdf(state[i, t] | stat, sig[i]);

        // generate posterior prediction for current trial
        y_pred[i, t] = normal_rng(stat, sig[i]);

        // update, retained state - learning rate * error
        stat = A[i]*stat - B[i]*(stat-rotation[i, t]);
      }
    }
  }
}