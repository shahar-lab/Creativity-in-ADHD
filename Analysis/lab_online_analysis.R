# ============================================================
# TESTING LOCATION ANALYSIS
# Lab vs Online
# ============================================================


# 1. Packages ----

library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(bayestestR)

set.seed(2026)

options(mc.cores = parallel::detectCores())


# 2. Load data ----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)


# 3. Define testing location ----

df <- df %>%
  mutate(
    
    testing_location = factor(
      Study_Location,
      levels = c("lab", "online"),
      labels = c("Lab", "Online")
    ),
    
    # Lab    = -0.5
    # Online = +0.5
    location_c = case_when(
      testing_location == "Lab"    ~ -0.5,
      testing_location == "Online" ~  0.5,
      TRUE                         ~ NA_real_
    )
  )

table(df$testing_location)

# 3.1 Originality

originality_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    n = n(),
    mean = mean(`Gallery Orig`, na.rm = TRUE),
    sd = sd(`Gallery Orig`, na.rm = TRUE),
    median = median(`Gallery Orig`, na.rm = TRUE),
    .groups = "drop"
  )

originality_location_descriptives

originality_center <- mean(
  df$`Gallery Orig`,
  na.rm = TRUE
)

originality_scale <- sd(
  df$`Gallery Orig`,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    originality_z =
      (`Gallery Orig` - originality_center) /
      originality_scale
  )

priors_originality_location <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)

m_originality_location <- brm(
  originality_z ~ location_c,
  
  data = df,
  family = student(),
  
  prior = priors_originality_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_originality_location)

originality_location_draws <- as_draws_df(
  m_originality_location
) %>%
  mutate(
    location_diff_orig =
      b_location_c * originality_scale
  )

originality_location_difference <- describe_posterior(
  originality_location_draws$location_diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

originality_location_difference

originality_location_draws <- originality_location_draws %>%
  mutate(
    
    lab_z =
      b_Intercept - 0.5 * b_location_c,
    
    online_z =
      b_Intercept + 0.5 * b_location_c,
    
    lab_originality =
      lab_z * originality_scale + originality_center,
    
    online_originality =
      online_z * originality_scale + originality_center
  )


originality_location_summary <- tibble(
  location = c("Lab", "Online"),
  
  posterior_median = c(
    median(originality_location_draws$lab_originality),
    median(originality_location_draws$online_originality)
  ),
  
  lower_90 = c(
    quantile(originality_location_draws$lab_originality, 0.05),
    quantile(originality_location_draws$online_originality, 0.05)
  ),
  
  upper_90 = c(
    quantile(originality_location_draws$lab_originality, 0.95),
    quantile(originality_location_draws$online_originality, 0.95)
  )
)

originality_location_summary

# ============================================================
# FLUENCY: Lab vs Online
# ============================================================

#. 4. Fluency - Lab vs Online

# 4.1 Create fluency variable ----

df <- df %>%
  mutate(
    fluency = `#galleries`
  )


# 4.2 Descriptives by testing location ----

fluency_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    n = n(),
    mean = mean(fluency, na.rm = TRUE),
    sd = sd(fluency, na.rm = TRUE),
    median = median(fluency, na.rm = TRUE),
    .groups = "drop"
  )

fluency_location_descriptives

# 4.3. Priors ----

priors_fluency_location <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(log(40), 0.5), class = "Intercept"),
  prior(exponential(0.5), class = "shape")
)


# 4.4 Bayesian Lab vs Online fluency model ----

m_fluency_location <- brm(
  fluency ~ location_c,
  
  data = df,
  family = negbinomial(link = "log"),
  
  prior = priors_fluency_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_fluency_location)

# 4.5 Posterior draws and back-transformation ----

fluency_location_draws <- as_draws_df(
  m_fluency_location
) %>%
  mutate(
    
    # Expected log-count
    lab_log =
      b_Intercept - 0.5 * b_location_c,
    
    online_log =
      b_Intercept + 0.5 * b_location_c,
    
    # Expected number of saved shapes
    lab_fluency =
      exp(lab_log),
    
    online_fluency =
      exp(online_log),
    
    # Online - Lab difference
    diff_fluency =
      online_fluency - lab_fluency,
    
    # Online / Lab rate ratio
    rate_ratio =
      exp(b_location_c)
  )

# 4.6. Posterior estimates by testing location ----

fluency_location_summary <- tibble(
  location = c("Lab", "Online"),
  
  posterior_median = c(
    median(fluency_location_draws$lab_fluency),
    median(fluency_location_draws$online_fluency)
  ),
  
  lower_90 = c(
    quantile(fluency_location_draws$lab_fluency, 0.05),
    quantile(fluency_location_draws$online_fluency, 0.05)
  ),
  
  upper_90 = c(
    quantile(fluency_location_draws$lab_fluency, 0.95),
    quantile(fluency_location_draws$online_fluency, 0.95)
  )
)

fluency_location_summary

# 4.7. Online - Lab difference ----

fluency_location_difference <- describe_posterior(
  fluency_location_draws$diff_fluency,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

fluency_location_difference

# ============================================================
# g: Lab vs Online
# ============================================================

# 5. g - Lab vs Online ----


# 5.1 Descriptives by testing location ----

g_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    n = sum(!is.na(g_empirical)),
    mean = mean(g_empirical, na.rm = TRUE),
    sd = sd(g_empirical, na.rm = TRUE),
    median = median(g_empirical, na.rm = TRUE),
    .groups = "drop"
  )

g_location_descriptives


# 5.2 Standardize g ----

g_center <- mean(
  df$g_empirical,
  na.rm = TRUE
)

g_scale <- sd(
  df$g_empirical,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    g_z =
      (g_empirical - g_center) /
      g_scale
  )


# 5.3 Priors for continuous location analyses ----

priors_continuous_location <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)


# 5.4 Bayesian Lab vs Online g model ----

m_g_location <- brm(
  g_z ~ location_c,
  
  data = df,
  family = student(),
  
  prior = priors_continuous_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_g_location)


# 5.5 Posterior draws ----

g_location_draws <- as_draws_df(
  m_g_location
) %>%
  mutate(
    
    # Standardized estimates by testing location
    lab_g =
      b_Intercept - 0.5 * b_location_c,
    
    online_g =
      b_Intercept + 0.5 * b_location_c,
    
    # Online - Lab difference in SD units
    diff_g =
      b_location_c,
    
    # Online - Lab difference in original PCA units
    diff_g_orig =
      b_location_c * g_scale
  )


# 5.6 Posterior estimates by testing location ----

g_location_summary <- tibble(
  location = c("Lab", "Online"),
  
  posterior_median = c(
    median(g_location_draws$lab_g),
    median(g_location_draws$online_g)
  ),
  
  lower_90 = c(
    quantile(g_location_draws$lab_g, 0.05),
    quantile(g_location_draws$online_g, 0.05)
  ),
  
  upper_90 = c(
    quantile(g_location_draws$lab_g, 0.95),
    quantile(g_location_draws$online_g, 0.95)
  )
)

g_location_summary


# 5.7 Online - Lab difference ----

g_location_difference <- describe_posterior(
  g_location_draws$diff_g,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

g_location_difference

# 5.8. Plot for g - lab vs online

location_colors <- c(
  "Lab" = "#7B61A8",     # purple
  "Online" = "#4DAF4A"   # green
)

g_location_plot_draws <- g_location_draws %>%
  select(lab_g, online_g) %>%
  pivot_longer(
    cols = everything(),
    names_to = "location",
    values_to = "g_value"
  ) %>%
  mutate(
    location = recode(
      location,
      lab_g = "Lab",
      online_g = "Online"
    ),
    location = factor(
      location,
      levels = c("Lab", "Online")
    )
  )

p_g_location_density <- ggplot(
  g_location_plot_draws,
  aes(
    x = g_value,
    fill = location,
    color = location
  )
) +
  geom_density(
    alpha = 0.45,
    linewidth = 1.4
  ) +
  
  geom_point(
    data = g_location_summary,
    aes(
      x = posterior_median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(values = location_colors) +
  scale_color_manual(values = location_colors) +
  
  labs(
    x = expression("Estimated standardized " * italic(g)),
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "right"
  )

p_g_location_density

ggsave(
  filename = "Figures/lab_online/g_location_density.png",
  plot = p_g_location_density,
  width = 8,
  height = 5,
  dpi = 300
)

# plot diff online_lab_g
p_g_location_diff <- ggplot(
  g_location_draws,
  aes(x = diff_g)
) +
  geom_density(
    fill = "#BDBDBD",
    color = "#5A5A5A",
    alpha = 0.7,
    linewidth = 1.3
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 1
  ) +
  geom_point(
    aes(
      x = median(diff_g),
      y = 0
    ),
    stat = "unique",
    color = "black",
    size = 3
  ) +
  labs(
    x = expression("Estimated Online - Lab difference in standardized " * italic(g)),
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
  )

p_g_location_diff

# 5.9 Save g difference plot ----

ggsave(
  filename = "Figures/lab_online/g_location_difference.png",
  plot = p_g_location_diff,
  width = 8,
  height = 5,
  dpi = 300
)

### ADHD Robustness Analysis ###

# 5.10 Define ADHD group for robustness analysis ----

df <- df %>%
  mutate(
    
    group = case_when(
      diva_group == "TD"   ~ "Without ADHD",
      diva_group == "ADHD" ~ "ADHD",
      TRUE                 ~ NA_character_
    ),
    
    group = factor(
      group,
      levels = c("Without ADHD", "ADHD")
    ),
    
    group_c = case_when(
      group == "Without ADHD" ~ -0.5,
      group == "ADHD"         ~  0.5,
      TRUE                    ~ NA_real_
    )
  )

table(df$testing_location, df$group)

# 5.11 Robustness model: Testing location + ADHD group ----

m_g_location_adhd_adjusted <- brm(
  g_z ~ location_c + group_c,
  
  data = df,
  family = student(),
  
  prior = priors_continuous_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_g_location_adhd_adjusted)

# 5.12 Posterior summary: adjusted model ----

g_location_adhd_adjusted_posterior <- describe_posterior(
  m_g_location_adhd_adjusted,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

g_location_adhd_adjusted_posterior

# 5.13 Robustness model: Testing location x ADHD group ----

m_g_location_adhd_interaction <- brm(
  g_z ~ location_c * group_c,
  
  data = df,
  family = student(),
  
  prior = priors_continuous_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_g_location_adhd_interaction)

# 5.14 Posterior summary: interaction model ----

g_location_adhd_interaction_posterior <- describe_posterior(
  m_g_location_adhd_interaction,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

g_location_adhd_interaction_posterior

### Alpha ###

# 6. Alpha - Lab vs Online ----


# 6.1 Descriptives by testing location ----

alpha_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    n = sum(!is.na(alpha_empirical)),
    mean = mean(alpha_empirical, na.rm = TRUE),
    sd = sd(alpha_empirical, na.rm = TRUE),
    median = median(alpha_empirical, na.rm = TRUE),
    .groups = "drop"
  )

alpha_location_descriptives

# 6.2 Standardize alpha ----

alpha_center <- mean(
  df$alpha_empirical,
  na.rm = TRUE
)

alpha_scale <- sd(
  df$alpha_empirical,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    alpha_z =
      (alpha_empirical - alpha_center) /
      alpha_scale
  )

# 6.3 Bayesian Lab vs Online alpha model ----

m_alpha_location <- brm(
  alpha_z ~ location_c,
  
  data = df,
  family = student(),
  
  prior = priors_continuous_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_alpha_location)

# 6.4 Posterior draws ----

alpha_location_draws <- as_draws_df(
  m_alpha_location
) %>%
  mutate(
    
    lab_alpha =
      b_Intercept - 0.5 * b_location_c,
    
    online_alpha =
      b_Intercept + 0.5 * b_location_c,
    
    # Online - Lab difference in SD units
    diff_alpha =
      b_location_c,
    
    # Difference in original PCA units
    diff_alpha_orig =
      b_location_c * alpha_scale
  )

# 6.5 Posterior estimates by testing location ----

alpha_location_summary <- tibble(
  location = c("Lab", "Online"),
  
  posterior_median = c(
    median(alpha_location_draws$lab_alpha),
    median(alpha_location_draws$online_alpha)
  ),
  
  lower_90 = c(
    quantile(alpha_location_draws$lab_alpha, 0.05),
    quantile(alpha_location_draws$online_alpha, 0.05)
  ),
  
  upper_90 = c(
    quantile(alpha_location_draws$lab_alpha, 0.95),
    quantile(alpha_location_draws$online_alpha, 0.95)
  )
)

alpha_location_summary

# 6.6 Online - Lab difference ----

alpha_location_difference <- describe_posterior(
  alpha_location_draws$diff_alpha,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

alpha_location_difference

# 7. Efficiency - Lab vs Online ----

# 7.1 Descriptives by testing location and phase ----

efficiency_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    
    exploration_mean = mean(`exp efficiency`, na.rm = TRUE),
    exploration_sd = sd(`exp efficiency`, na.rm = TRUE),
    exploration_median = median(`exp efficiency`, na.rm = TRUE),
    
    exploitation_mean = mean(`scav efficiency`, na.rm = TRUE),
    exploitation_sd = sd(`scav efficiency`, na.rm = TRUE),
    exploitation_median = median(`scav efficiency`, na.rm = TRUE),
    
    .groups = "drop"
  )

efficiency_location_descriptives

# 7.2 Prepare efficiency data in long format ----

efficiency_location_long <- df %>%
  select(
    ID,
    testing_location,
    location_c,
    `exp efficiency`,
    `scav efficiency`
  ) %>%
  
  pivot_longer(
    cols = c(
      `exp efficiency`,
      `scav efficiency`
    ),
    names_to = "phase",
    values_to = "efficiency"
  ) %>%
  
  mutate(
    
    phase = case_when(
      phase == "exp efficiency"  ~ "Exploration",
      phase == "scav efficiency" ~ "Exploitation"
    ),
    
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    ),
    
    phase_c = case_when(
      phase == "Exploration"  ~ -0.5,
      phase == "Exploitation" ~  0.5
    )
  )

table(
  efficiency_location_long$testing_location,
  efficiency_location_long$phase
)

# 7.3 Standardize efficiency ----

efficiency_location_center <- mean(
  efficiency_location_long$efficiency,
  na.rm = TRUE
)

efficiency_location_scale <- sd(
  efficiency_location_long$efficiency,
  na.rm = TRUE
)

efficiency_location_long <- efficiency_location_long %>%
  mutate(
    efficiency_z =
      (efficiency - efficiency_location_center) /
      efficiency_location_scale
  )

# 7.4 Priors ----

priors_efficiency_location <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(exponential(1), class = "sd"),
  prior(gamma(2, 0.1), class = "nu")
)

# 7.5 Bayesian Testing Location x Phase efficiency model ----

m_efficiency_location <- brm(
  efficiency_z ~ location_c * phase_c + (1 | ID),
  
  data = efficiency_location_long,
  family = student(),
  
  prior = priors_efficiency_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_efficiency_location)

# 7.6 Posterior summary ----

efficiency_location_posterior <- describe_posterior(
  m_efficiency_location,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_location_posterior

# 7.7 Back-transform effects to original efficiency units ----

efficiency_location_draws <- as_draws_df(
  m_efficiency_location
) %>%
  mutate(
    
    # Online - Lab
    location_diff_orig =
      b_location_c * efficiency_location_scale,
    
    # Exploitation - Exploration
    phase_diff_orig =
      b_phase_c * efficiency_location_scale,
    
    # Location x Phase interaction
    interaction_orig =
      `b_location_c:phase_c` * efficiency_location_scale
  )


efficiency_location_difference <- describe_posterior(
  efficiency_location_draws$location_diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_phase_difference_location_model <- describe_posterior(
  efficiency_location_draws$phase_diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_location_interaction <- describe_posterior(
  efficiency_location_draws$interaction_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

# 7.8 Online - Lab difference separately by phase ----

efficiency_location_draws <- efficiency_location_draws %>%
  mutate(
    
    # Online - Lab during Exploration
    location_diff_exploration =
      (b_location_c -
         0.5 * `b_location_c:phase_c`) *
      efficiency_location_scale,
    
    # Online - Lab during Exploitation
    location_diff_exploitation =
      (b_location_c +
         0.5 * `b_location_c:phase_c`) *
      efficiency_location_scale
  )

efficiency_location_diff_exploration <- describe_posterior(
  efficiency_location_draws$location_diff_exploration,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_location_diff_exploitation <- describe_posterior(
  efficiency_location_draws$location_diff_exploitation,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_location_diff_exploration
efficiency_location_diff_exploitation

# 7.9 Robustness model: Testing location x Phase + ADHD group ----

m_efficiency_location_adhd_adjusted <- brm(
  efficiency_z ~ location_c * phase_c + group_c + (1 | ID),
  
  data = efficiency_location_long %>%
    left_join(
      df %>% select(ID, group_c),
      by = "ID"
    ),
  
  family = student(),
  
  prior = priors_efficiency_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

# 7.10 Posterior summary: ADHD-adjusted efficiency model ----

efficiency_location_adhd_adjusted_posterior <- describe_posterior(
  m_efficiency_location_adhd_adjusted,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_location_adhd_adjusted_posterior

# 7.11 Posterior estimates by testing location and phase ----

efficiency_cell_draws <- as_draws_df(
  m_efficiency_location
) %>%
  as_tibble() %>%
  transmute(
    
    Lab_Exploration =
      (
        b_Intercept
        - 0.5 * b_location_c
        - 0.5 * b_phase_c
        + 0.25 * `b_location_c:phase_c`
      ) * efficiency_location_scale +
      efficiency_location_center,
    
    Lab_Exploitation =
      (
        b_Intercept
        - 0.5 * b_location_c
        + 0.5 * b_phase_c
        - 0.25 * `b_location_c:phase_c`
      ) * efficiency_location_scale +
      efficiency_location_center,
    
    Online_Exploration =
      (
        b_Intercept
        + 0.5 * b_location_c
        - 0.5 * b_phase_c
        - 0.25 * `b_location_c:phase_c`
      ) * efficiency_location_scale +
      efficiency_location_center,
    
    Online_Exploitation =
      (
        b_Intercept
        + 0.5 * b_location_c
        + 0.5 * b_phase_c
        + 0.25 * `b_location_c:phase_c`
      ) * efficiency_location_scale +
      efficiency_location_center
  )

efficiency_cell_summary <- efficiency_cell_draws %>%
  pivot_longer(
    cols = everything(),
    names_to = "cell",
    values_to = "efficiency"
  ) %>%
  separate(
    cell,
    into = c("location", "phase"),
    sep = "_"
  ) %>%
  group_by(location, phase) %>%
  summarise(
    posterior_median = median(efficiency),
    lower_90 = quantile(efficiency, 0.05),
    upper_90 = quantile(efficiency, 0.95),
    .groups = "drop"
  ) %>%
  mutate(
    location = factor(location, levels = c("Lab", "Online")),
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    )
  )

efficiency_cell_summary

### Plots efficiency ###

location_colors <- c(
  "Lab" = "#7B61A8",
  "Online" = "#4DAF4A"
)

# 7.12 Plot efficiency by phase and testing location ----

efficiency_cell_summary <- efficiency_cell_summary %>%
  mutate(
    x_position = case_when(
      phase == "Exploration" & location == "Lab" ~ 0.94,
      phase == "Exploration" & location == "Online" ~ 1.06,
      phase == "Exploitation" & location == "Lab" ~ 1.94,
      phase == "Exploitation" & location == "Online" ~ 2.06
    )
  )

p_efficiency_location <- ggplot(
  efficiency_cell_summary,
  aes(
    x = x_position,
    y = posterior_median,
    color = location,
    group = location
  )
) +
  
  geom_line(
    linewidth = 1.2,
    alpha = 0.8
  ) +
  
  geom_errorbar(
    aes(
      ymin = lower_90,
      ymax = upper_90
    ),
    width = 0.04,
    linewidth = 1
  ) +
  
  geom_point(
    size = 4
  ) +
  
  scale_x_continuous(
    breaks = c(1, 2),
    labels = c("Exploration", "Exploitation"),
    limits = c(0.75, 2.25)
  ) +
  
  scale_color_manual(
    values = location_colors
  ) +
  
  labs(
    x = NULL,
    y = "Estimated search efficiency",
    color = NULL
  ) +
  
  theme_classic(base_size = 16) +
  
  theme(
    legend.position = "right"
  )

ggsave(
  "Figures/lab_online/efficiency_location_by_phase.png",
  plot = p_efficiency_location,
  width = 8,
  height = 5,
  dpi = 300
)

ggsave(
  "Figures/lab_online/efficiency_location_by_phase.pdf",
  plot = p_efficiency_location,
  width = 8,
  height = 5
)

p_efficiency_location

# 7.13 Posterior Online - Lab difference by phase ----

efficiency_phase_diff_plot_draws <- efficiency_location_draws %>%
  as_tibble() %>%
  select(
    location_diff_exploration,
    location_diff_exploitation
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "phase",
    values_to = "difference"
  ) %>%
  mutate(
    phase = recode(
      phase,
      location_diff_exploration = "Exploration",
      location_diff_exploitation = "Exploitation"
    ),
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    )
  )

phase_colors <- c(
  "Exploration" = "#E69F00",
  "Exploitation" = "#009E73"
)

p_efficiency_location_diff_phase <- ggplot(
  efficiency_phase_diff_plot_draws,
  aes(
    x = difference,
    fill = phase,
    color = phase
  )
) +
  
  geom_density(
    alpha = 0.40,
    linewidth = 1.3
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 1
  ) +
  
  scale_fill_manual(values = phase_colors) +
  scale_color_manual(values = phase_colors) +
  
  labs(
    x = "Online - Lab difference in search efficiency",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 16) +
  
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    legend.position = "right"
  )

p_efficiency_location_diff_phase

ggsave(
  "Figures/lab_online/efficiency_location_difference_by_phase.png",
  plot = p_efficiency_location_diff_phase,
  width = 8,
  height = 5,
  dpi = 300
)

# ============================================================
# 8. TASK DURATION: Lab vs Online
# ============================================================

# 8.1 Descriptives ----

duration_location_descriptives <- df %>%
  group_by(testing_location) %>%
  summarise(
    n = sum(!is.na(`Total Play Time`)),
    mean = mean(`Total Play Time`, na.rm = TRUE),
    sd = sd(`Total Play Time`, na.rm = TRUE),
    median = median(`Total Play Time`, na.rm = TRUE),
    q1 = quantile(`Total Play Time`, 0.25, na.rm = TRUE),
    q3 = quantile(`Total Play Time`, 0.75, na.rm = TRUE),
    min = min(`Total Play Time`, na.rm = TRUE),
    max = max(`Total Play Time`, na.rm = TRUE),
    .groups = "drop"
  )

duration_location_descriptives

# 8.2 Standardize task duration ----

duration_center <- mean(
  df$`Total Play Time`,
  na.rm = TRUE
)

duration_scale <- sd(
  df$`Total Play Time`,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    duration_z =
      (`Total Play Time` - duration_center) /
      duration_scale
  )

# 8.3 Priors ----

priors_duration_location <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)

# 8.4 Bayesian Lab vs Online task-duration model ----

m_duration_location <- brm(
  duration_z ~ location_c,
  
  data = df,
  family = student(),
  
  prior = priors_duration_location,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_duration_location)

# 8.5 Posterior draws and back-transformation ----

duration_location_draws <- as_draws_df(
  m_duration_location
) %>%
  mutate(
    
    lab_z =
      b_Intercept - 0.5 * b_location_c,
    
    online_z =
      b_Intercept + 0.5 * b_location_c,
    
    lab_seconds =
      lab_z * duration_scale + duration_center,
    
    online_seconds =
      online_z * duration_scale + duration_center,
    
    # Online - Lab
    diff_z = b_location_c,
    
    diff_seconds =
      b_location_c * duration_scale
  )

# 8.6 Posterior estimates by testing location ----

duration_location_summary <- tibble(
  location = c("Lab", "Online"),
  
  posterior_median = c(
    median(duration_location_draws$lab_seconds),
    median(duration_location_draws$online_seconds)
  ),
  
  lower_90 = c(
    quantile(duration_location_draws$lab_seconds, 0.05),
    quantile(duration_location_draws$online_seconds, 0.05)
  ),
  
  upper_90 = c(
    quantile(duration_location_draws$lab_seconds, 0.95),
    quantile(duration_location_draws$online_seconds, 0.95)
  )
)

duration_location_summary

# Online - Lab difference in seconds

duration_location_difference <- describe_posterior(
  duration_location_draws$diff_seconds,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

duration_location_difference