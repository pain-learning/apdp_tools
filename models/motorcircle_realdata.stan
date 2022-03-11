// motor circle model real-data
data {
  int<lower=1> N; // number of participants
  int<lower=1> T; // number of trials
  // int<lower=1, upper=T> Tsubj[N];
  real x_target; // task param target circle x
  real y_target; // task param target circle y
  real x_penalty; // task param penalty circle x
  real y_penalty; // task param penalty circle y
  real radius; // task param target circle radius
  real penalty_val; // task param penalty value

  real x[N, T]; // observed x
  real y[N, T]; // observed y
}
parameters {
  real<lower=0> loss_sens; // loss sensitivity
  real<lower=0> perturb; // covariance
}
transformed parameters {
  // real x_sj = x_target + (-radius*penalty_val/100)*loss_sens; // subject version of x
  // real sigx = perturb * (x_target-x_penalty); //subject covx
  // real sigy = perturb * (x_target-x_penalty); //subject covx
  vector[2] xy_sj = [x_target, y_target + (-radius*penalty_val)*loss_sens]'; // subject version of x
  // matrix[2,2] sig = [[perturb * (x_target-x_penalty), 0], [0, perturb * (x_target-x_penalty)]]';
  cov_matrix[2] sig = [[perturb * (y_target-y_penalty), 0], [0, perturb * (y_target-y_penalty)]]';

}
model {
  loss_sens ~ normal(0,1);
  perturb ~ normal(0,1);

  for (i in 1:N) {
    for (t in 1:T) {
      // target += normal_lpdf(x[i, t] | x_sj, sigx);       // prior log-density
      vector[2] xy = [x[i,t], y[i,t]]'; // subject version of x 
      target += multi_normal_lpdf(xy | xy_sj, sig);       // prior log-density
    };
  };
}