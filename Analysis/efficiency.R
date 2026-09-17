# ============================================================
# SEARCH EFFICIENCY
# ADHD vs Without ADHD
# ============================================================


# 1. Packages ----

library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(bayestestR)

set.seed(2026)

options(mc.cores = parallel::detectCores())


# 2. Load data ----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)


# 3. Define analysis groups ----

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

table(df$group)

# 4. EFFICIENCY: Initial descriptives ----

efficiency_descriptives <- df %>%
  group_by(group) %>%
  summarise(
    
    exploration_mean = mean(`exp efficiency`, na.rm = TRUE),
    exploration_sd = sd(`exp efficiency`, na.rm = TRUE),
    exploration_median = median(`exp efficiency`, na.rm = TRUE),
    
    exploitation_mean = mean(`scav efficiency`, na.rm = TRUE),
    exploitation_sd = sd(`scav efficiency`, na.rm = TRUE),
    exploitation_median = median(`scav efficiency`, na.rm = TRUE),
    
    .groups = "drop"
  )

efficiency_descriptives

# 4.1 Check efficiency ranges ----

efficiency_ranges <- df %>%
  summarise(
    exp_min = min(`exp efficiency`, na.rm = TRUE),
    exp_max = max(`exp efficiency`, na.rm = TRUE),
    scav_min = min(`scav efficiency`, na.rm = TRUE),
    scav_max = max(`scav efficiency`, na.rm = TRUE),
    
    exp_zero = sum(`exp efficiency` == 0, na.rm = TRUE),
    exp_one  = sum(`exp efficiency` == 1, na.rm = TRUE),
    
    scav_zero = sum(`scav efficiency` == 0, na.rm = TRUE),
    scav_one  = sum(`scav efficiency` == 1, na.rm = TRUE)
  )

efficiency_ranges

# 5. Prepare efficiency data in long format ----

efficiency_long <- df %>%
  select(
    ID,
    group,
    group_c,
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
    
    # Exploration  = -0.5
    # Exploitation = +0.5
    phase_c = case_when(
      phase == "Exploration"  ~ -0.5,
      phase == "Exploitation" ~  0.5
    )
  )

# 6. Standardize efficiency ----

efficiency_center <- mean(
  efficiency_long$efficiency,
  na.rm = TRUE
)

efficiency_scale <- sd(
  efficiency_long$efficiency,
  na.rm = TRUE
)

efficiency_long <- efficiency_long %>%
  mutate(
    efficiency_z =
      (efficiency - efficiency_center) / efficiency_scale
  )

# 7. Priors for efficiency model ----

priors_efficiency <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(exponential(1), class = "sd"),
  prior(gamma(2, 0.1), class = "nu")
)

# 8. Bayesian Group x Phase efficiency model ----

m_efficiency <- brm(
  efficiency_z ~ group_c * phase_c + (1 | ID),
  
  data = efficiency_long,
  family = student(),
  
  prior = priors_efficiency,
  
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

summary(m_efficiency)

# 9. Posterior summary ----

efficiency_posterior <- describe_posterior(
  m_efficiency,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_posterior

# 10. Back-transform efficiency effects to original units ----

efficiency_draws <- as_draws_df(m_efficiency) %>%
  mutate(
    
    # Exploitation - Exploration difference
    phase_diff_orig =
      b_phase_c * efficiency_scale,
    
    # ADHD - Without ADHD difference
    group_diff_orig =
      b_group_c * efficiency_scale,
    
    # Group x Phase interaction
    interaction_orig =
      `b_group_c:phase_c` * efficiency_scale
  )


# Phase difference
efficiency_phase_difference <- describe_posterior(
  efficiency_draws$phase_diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

# Group difference
efficiency_group_difference <- describe_posterior(
  efficiency_draws$group_diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

# Group x Phase interaction
efficiency_interaction <- describe_posterior(
  efficiency_draws$interaction_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

efficiency_phase_difference
efficiency_group_difference
efficiency_interaction

# 11. Posterior efficiency estimates for each group ----

efficiency_draws <- efficiency_draws %>%
  mutate(
    
    # Group estimates averaged across phases
    without_adhd_eff_z =
      b_Intercept - 0.5 * b_group_c,
    
    adhd_eff_z =
      b_Intercept + 0.5 * b_group_c,
    
    # Back-transform to original efficiency scale
    without_adhd_eff =
      without_adhd_eff_z * efficiency_scale + efficiency_center,
    
    adhd_eff =
      adhd_eff_z * efficiency_scale + efficiency_center
  )

efficiency_group_summary <- tibble(
  group = c("Without ADHD", "ADHD"),
  
  posterior_median = c(
    median(efficiency_draws$without_adhd_eff),
    median(efficiency_draws$adhd_eff)
  ),
  
  lower_90 = c(
    quantile(efficiency_draws$without_adhd_eff, 0.05),
    quantile(efficiency_draws$adhd_eff, 0.05)
  ),
  
  upper_90 = c(
    quantile(efficiency_draws$without_adhd_eff, 0.95),
    quantile(efficiency_draws$adhd_eff, 0.95)
  )
)

efficiency_group_summary

# ADHD PRESENTATION ANALYSIS: SEARCH EFFICIENCY ----


# 1. Prepare long-format data ----

efficiency_presentation_long <- df %>%
  select(
    ID,
    ADHD_subtype,
    `exp efficiency`,
    `scav efficiency`
  ) %>%
  
  mutate(
    presentation3 = factor(
      ADHD_subtype,
      levels = c(
        "none",
        "inattentive",
        "combined"
      ),
      labels = c(
        "Without ADHD",
        "Inattentive",
        "Combined/HI"
      )
    )
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
  efficiency_presentation_long$presentation3,
  efficiency_presentation_long$phase
)

# 2. Standardize using the same scale as the primary analysis ----

efficiency_presentation_long <- efficiency_presentation_long %>%
  mutate(
    efficiency_z =
      (efficiency - efficiency_center) / efficiency_scale
  )

# 3. Bayesian model: efficiency across ADHD presentations ----

m_efficiency_presentation <- brm(
  efficiency_z ~ presentation3 + phase_c + (1 | ID),
  
  data = efficiency_presentation_long,
  family = student(),
  
  prior = priors_efficiency,
  
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

summary(m_efficiency_presentation)

# 4. Posterior efficiency estimates by presentation ----

newdata_efficiency_presentation <- tibble(
  presentation3 = factor(
    c(
      "Without ADHD",
      "Inattentive",
      "Combined/HI"
    ),
    levels = levels(
      efficiency_presentation_long$presentation3
    )
  ),
  
  phase_c = 0
)

efficiency_presentation_epred_z <- posterior_epred(
  m_efficiency_presentation,
  newdata = newdata_efficiency_presentation,
  re_formula = NA
)

# Back-transform to original efficiency units
efficiency_presentation_epred <-
  efficiency_presentation_epred_z *
  efficiency_scale +
  efficiency_center

# 5. Posterior summary by presentation ----

efficiency_presentation_summary <- tibble(
  group = c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  ),
  
  posterior_median = apply(
    efficiency_presentation_epred,
    2,
    median
  ),
  
  lower_90 = apply(
    efficiency_presentation_epred,
    2,
    quantile,
    probs = 0.05
  ),
  
  upper_90 = apply(
    efficiency_presentation_epred,
    2,
    quantile,
    probs = 0.95
  )
)

efficiency_presentation_summary

# 6. Pairwise posterior contrasts ----

eff_diff_inatt_vs_without <-
  efficiency_presentation_epred[, 2] -
  efficiency_presentation_epred[, 1]

eff_diff_combined_vs_without <-
  efficiency_presentation_epred[, 3] -
  efficiency_presentation_epred[, 1]

eff_diff_combined_vs_inatt <-
  efficiency_presentation_epred[, 3] -
  efficiency_presentation_epred[, 2]


efficiency_presentation_contrasts <- tibble(
  contrast = c(
    "Inattentive - Without ADHD",
    "Combined/HI - Without ADHD",
    "Combined/HI - Inattentive"
  ),
  
  median = c(
    median(eff_diff_inatt_vs_without),
    median(eff_diff_combined_vs_without),
    median(eff_diff_combined_vs_inatt)
  ),
  
  lower_90 = c(
    quantile(eff_diff_inatt_vs_without, 0.05),
    quantile(eff_diff_combined_vs_without, 0.05),
    quantile(eff_diff_combined_vs_inatt, 0.05)
  ),
  
  upper_90 = c(
    quantile(eff_diff_inatt_vs_without, 0.95),
    quantile(eff_diff_combined_vs_without, 0.95),
    quantile(eff_diff_combined_vs_inatt, 0.95)
  ),
  
  pd = c(
    max(
      mean(eff_diff_inatt_vs_without > 0),
      mean(eff_diff_inatt_vs_without < 0)
    ),
    
    max(
      mean(eff_diff_combined_vs_without > 0),
      mean(eff_diff_combined_vs_without < 0)
    ),
    
    max(
      mean(eff_diff_combined_vs_inatt > 0),
      mean(eff_diff_combined_vs_inatt < 0)
    )
  ) * 100
)

efficiency_presentation_contrasts

# 12. Posterior efficiency estimates by Group x Phase ----

newdata_efficiency <- tibble(
  group = factor(
    c(
      "Without ADHD",
      "ADHD",
      "Without ADHD",
      "ADHD"
    ),
    levels = levels(df$group)
  ),
  
  group_c = c(
    -0.5,
    0.5,
    -0.5,
    0.5
  ),
  
  phase = factor(
    c(
      "Exploration",
      "Exploration",
      "Exploitation",
      "Exploitation"
    ),
    levels = c("Exploration", "Exploitation")
  ),
  
  phase_c = c(
    -0.5,
    -0.5,
    0.5,
    0.5
  )
)

efficiency_epred_z <- posterior_epred(
  m_efficiency,
  newdata = newdata_efficiency,
  re_formula = NA
)

# Back-transform to original efficiency scale
efficiency_epred <-
  efficiency_epred_z * efficiency_scale +
  efficiency_center

# 13. Phase-specific ADHD - Without ADHD contrasts ----

eff_diff_exploration <-
  efficiency_epred[, 2] -
  efficiency_epred[, 1]

eff_diff_exploitation <-
  efficiency_epred[, 4] -
  efficiency_epred[, 3]


efficiency_phase_group_contrasts <- tibble(
  phase = c("Exploration", "Exploitation"),
  
  median = c(
    median(eff_diff_exploration),
    median(eff_diff_exploitation)
  ),
  
  lower_90 = c(
    quantile(eff_diff_exploration, 0.05),
    quantile(eff_diff_exploitation, 0.05)
  ),
  
  upper_90 = c(
    quantile(eff_diff_exploration, 0.95),
    quantile(eff_diff_exploitation, 0.95)
  ),
  
  pd = c(
    max(
      mean(eff_diff_exploration > 0),
      mean(eff_diff_exploration < 0)
    ),
    
    max(
      mean(eff_diff_exploitation > 0),
      mean(eff_diff_exploitation < 0)
    )
  ) * 100
)

efficiency_phase_group_contrasts

# 14. FIGURE: Overall efficiency by phase ----

overall_phase_draws <- efficiency_draws %>%
  transmute(
    
    Exploration =
      (b_Intercept - 0.5 * b_phase_c) *
      efficiency_scale + efficiency_center,
    
    Exploitation =
      (b_Intercept + 0.5 * b_phase_c) *
      efficiency_scale + efficiency_center
  ) %>%
  
  pivot_longer(
    cols = everything(),
    names_to = "phase",
    values_to = "efficiency"
  ) %>%
  
  mutate(
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    )
  )


overall_phase_summary <- overall_phase_draws %>%
  group_by(phase) %>%
  summarise(
    median = median(efficiency),
    lower_90 = quantile(efficiency, 0.05),
    upper_90 = quantile(efficiency, 0.95),
    .groups = "drop"
  )

overall_phase_summary

# 14. FIGURE: Overall efficiency by phase ----

overall_phase_draws <- efficiency_draws %>%
  transmute(
    
    Exploration =
      (b_Intercept - 0.5 * b_phase_c) *
      efficiency_scale + efficiency_center,
    
    Exploitation =
      (b_Intercept + 0.5 * b_phase_c) *
      efficiency_scale + efficiency_center
  ) %>%
  
  pivot_longer(
    cols = everything(),
    names_to = "phase",
    values_to = "efficiency"
  ) %>%
  
  mutate(
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    )
  )


overall_phase_summary <- overall_phase_draws %>%
  group_by(phase) %>%
  summarise(
    median = median(efficiency),
    lower_90 = quantile(efficiency, 0.05),
    upper_90 = quantile(efficiency, 0.95),
    .groups = "drop"
  )

overall_phase_summary

# Plot: Overall efficiency by phase ----

p_eff_phase <- ggplot(
  overall_phase_draws,
  aes(
    x = phase,
    y = efficiency,
    fill = phase
  )
) +
  
  geom_violin(
    trim = FALSE,
    alpha = 0.60,
    color = NA
  ) +
  
  geom_errorbar(
    data = overall_phase_summary,
    aes(
      x = phase,
      ymin = lower_90,
      ymax = upper_90
    ),
    inherit.aes = FALSE,
    width = 0.07,
    color = "black",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = overall_phase_summary,
    aes(
      x = phase,
      y = median
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    values = c(
      "Exploration" = "#E69F00",
      "Exploitation" = "#009E73"
    )
  ) +
  
  labs(
    x = NULL,
    y = "Estimated search efficiency"
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    legend.position = "none"
  )

p_eff_phase

ggsave(
  "Figures/efficiency/efficiency_phase.png",
  plot = p_eff_phase,
  width = 7,
  height = 5,
  dpi = 300
)

# 15. FIGURE: Overall efficiency by ADHD group ----

overall_group_draws <- tibble(
  
  `Without ADHD` = efficiency_draws$without_adhd_eff,
  ADHD = efficiency_draws$adhd_eff
  
) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "efficiency"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c("Without ADHD", "ADHD")
    )
  )


# Posterior summary
overall_group_summary <- overall_group_draws %>%
  group_by(group) %>%
  summarise(
    median = median(efficiency),
    lower_90 = quantile(efficiency, 0.05),
    upper_90 = quantile(efficiency, 0.95),
    .groups = "drop"
  )

overall_group_summary

# FIGURE: Posterior distributions of overall efficiency by ADHD group ----

overall_group_draws <- overall_group_draws %>%
  mutate(
    group = factor(
      group,
      levels = c("ADHD", "Without ADHD")
    )
  )

group_colors <- c(
  "ADHD" = "#0072B2",
  "Without ADHD" = "#CC79A7"
)

p_eff_group <- ggplot(
  overall_group_draws,
  aes(
    x = efficiency,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.55,
    linewidth = 1.1
  ) +
  
  # baseline
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 1
  ) +
  
  # posterior medians
  geom_point(
    data = overall_group_summary,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 4
  ) +
  
  scale_fill_manual(
    values = group_colors,
    breaks = c("ADHD", "Without ADHD")
  ) +
  
  scale_color_manual(
    values = group_colors,
    breaks = c("ADHD", "Without ADHD")
  ) +
  
  labs(
    x = "Estimated overall search efficiency",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 18) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    legend.position = "right",
    legend.title = element_blank()
  )

p_eff_group

ggsave(
  "Figures/efficiency/efficiency_posterior_groups.png",
  plot = p_eff_group,
  width = 8,
  height = 5,
  dpi = 300
)

ggsave(
  "Figures/efficiency/efficiency_posterior_groups.pdf",
  plot = p_eff_group,
  width = 8,
  height = 5
)

# 16. Posterior distribution of overall group difference in efficiency ----

overall_efficiency_diff_draws <- efficiency_draws %>%
  transmute(
    without_adhd =
      (b_Intercept - 0.5 * b_group_c) *
      efficiency_scale + efficiency_center,
    
    adhd =
      (b_Intercept + 0.5 * b_group_c) *
      efficiency_scale + efficiency_center,
    
    diff = adhd - without_adhd
  )

overall_efficiency_diff_summary <- overall_efficiency_diff_draws %>%
  summarise(
    median = median(diff),
    lower_90 = quantile(diff, 0.05),
    upper_90 = quantile(diff, 0.95),
    pd = max(mean(diff > 0), mean(diff < 0)) * 100
  )

overall_efficiency_diff_summary

p_eff_diff <- ggplot(
  overall_efficiency_diff_draws,
  aes(x = diff)
) +
  
  geom_density(
    fill = "grey80",
    color = "grey35",
    linewidth = 1.2,
    alpha = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 1
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 1
  ) +
  
  geom_point(
    data = overall_efficiency_diff_summary,
    aes(x = median, y = 0),
    inherit.aes = FALSE,
    color = "black",
    size = 4
  ) +
  
  labs(
    x = "Estimated ADHD - Without ADHD difference in overall search efficiency",
    y = NULL
  ) +
  
  theme_classic(base_size = 18) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank()
  )

p_eff_diff

ggsave(
  "Figures/efficiency/efficiency_posterior_difference.png",
  plot = p_eff_diff,
  width = 8,
  height = 5,
  dpi = 300
)

ggsave(
  "Figures/efficiency/efficiency_posterior_difference.pdf",
  plot = p_eff_diff,
  width = 8,
  height = 5
)

###---------------###

# 17 Bayesian model: ADHD presentation x phase ----

m_efficiency_presentation_phase <- brm(
  efficiency_z ~ presentation3 * phase_c + (1 | ID),
  
  data = efficiency_presentation_long,
  family = student(),
  
  prior = priors_efficiency,
  
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

summary(m_efficiency_presentation_phase)

# 17.1 Posterior estimates by presentation and phase ----

newdata_efficiency_presentation_phase <- tibble(
  presentation3 = factor(
    rep(
      c("Without ADHD", "Inattentive", "Combined/HI"),
      times = 2
    ),
    levels = levels(efficiency_presentation_long$presentation3)
  ),
  
  phase = factor(
    rep(
      c("Exploration", "Exploitation"),
      each = 3
    ),
    levels = c("Exploration", "Exploitation")
  ),
  
  phase_c = rep(
    c(-0.5, 0.5),
    each = 3
  )
)

efficiency_presentation_phase_epred_z <- posterior_epred(
  m_efficiency_presentation_phase,
  newdata = newdata_efficiency_presentation_phase,
  re_formula = NA
)

# Back-transform to original efficiency units
efficiency_presentation_phase_epred <-
  efficiency_presentation_phase_epred_z *
  efficiency_scale +
  efficiency_center

# 17.2 Combined/HI - Inattentive contrast within each phase ----

eff_subtype_diff_exploration <-
  efficiency_presentation_phase_epred[, 3] -
  efficiency_presentation_phase_epred[, 2]

eff_subtype_diff_exploitation <-
  efficiency_presentation_phase_epred[, 6] -
  efficiency_presentation_phase_epred[, 5]

# 17.3 Posterior summaries of subtype differences by phase ----

eff_subtype_phase_contrasts <- tibble(
  phase = c("Exploration", "Exploitation"),
  
  median = c(
    median(eff_subtype_diff_exploration),
    median(eff_subtype_diff_exploitation)
  ),
  
  lower_90 = c(
    quantile(eff_subtype_diff_exploration, 0.05),
    quantile(eff_subtype_diff_exploitation, 0.05)
  ),
  
  upper_90 = c(
    quantile(eff_subtype_diff_exploration, 0.95),
    quantile(eff_subtype_diff_exploitation, 0.95)
  ),
  
  pd = c(
    max(
      mean(eff_subtype_diff_exploration > 0),
      mean(eff_subtype_diff_exploration < 0)
    ),
    
    max(
      mean(eff_subtype_diff_exploitation > 0),
      mean(eff_subtype_diff_exploitation < 0)
    )
  ) * 100
)

eff_subtype_phase_contrasts

# 17.4 Prepare posterior distributions for all ADHD presentations ----

# Exploration
eff_exploration_presentations <- 
  as_tibble(efficiency_presentation_phase_epred[, 1:3]) %>%
  setNames(c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  )) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "efficiency"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "Without ADHD",
        "Inattentive",
        "Combined/HI"
      )
    )
  )

# Exploitation
eff_exploitation_presentations <- 
  as_tibble(efficiency_presentation_phase_epred[, 4:6]) %>%
  setNames(c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  )) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "efficiency"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "Without ADHD",
        "Inattentive",
        "Combined/HI"
      )
    )
  )

# 17.5 Posterior medians ----

eff_exploration_medians <- eff_exploration_presentations %>%
  group_by(group) %>%
  summarise(
    median = median(efficiency),
    .groups = "drop"
  )

eff_exploitation_medians <- eff_exploitation_presentations %>%
  group_by(group) %>%
  summarise(
    median = median(efficiency),
    .groups = "drop"
  )

# 17.6 Common x-axis limits ----

eff_presentation_x_limits <- range(
  c(
    eff_exploration_presentations$efficiency,
    eff_exploitation_presentations$efficiency
  ),
  na.rm = TRUE
)

# 17.7 Plot: Exploration efficiency by ADHD presentation ----

p_eff_exploration_presentation <- ggplot(
  eff_exploration_presentations,
  aes(
    x = efficiency,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = eff_exploration_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  coord_cartesian(
    xlim = eff_presentation_x_limits
  ) +
  
  labs(
    x = "Estimated exploration efficiency",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_eff_exploration_presentation

# 17.8 Plot: Exploitation efficiency by ADHD presentation ----

p_eff_exploitation_presentation <- ggplot(
  eff_exploitation_presentations,
  aes(
    x = efficiency,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = eff_exploitation_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  coord_cartesian(
    xlim = eff_presentation_x_limits
  ) +
  
  labs(
    x = "Estimated exploitation efficiency",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_eff_exploitation_presentation

# 17.9 Save presentation plots ----

ggsave(
  filename = "Figures/efficiency/efficiency_exploration_by_ADHD_presentation.png",
  plot = p_eff_exploration_presentation,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = "Figures/efficiency/efficiency_exploitation_by_ADHD_presentation.png",
  plot = p_eff_exploitation_presentation,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)