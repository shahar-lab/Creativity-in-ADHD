# ============================================================
# FLUENCY: ADHD vs Without ADHD
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
    
    # Contrast coding:
    # Without ADHD = -0.5
    # ADHD         = +0.5
    #
    # Positive coefficient = ADHD > Without ADHD
    group_c = case_when(
      group == "Without ADHD" ~ -0.5,
      group == "ADHD"         ~  0.5,
      TRUE                    ~ NA_real_
    )
  )


# Check groups
table(df$group, useNA = "ifany")


# 4. Fluency descriptives and distribution ----

## 4.1 Fluency descriptives by group ----

fluency_descriptives <- df %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(`#galleries`, na.rm = TRUE),
    sd = sd(`#galleries`, na.rm = TRUE),
    median = median(`#galleries`, na.rm = TRUE),
    q1 = quantile(`#galleries`, 0.25, na.rm = TRUE),
    q3 = quantile(`#galleries`, 0.75, na.rm = TRUE),
    min = min(`#galleries`, na.rm = TRUE),
    max = max(`#galleries`, na.rm = TRUE),
    .groups = "drop"
  )

fluency_descriptives


## 4.2 Inspect fluency count distribution ----

fluency_distribution <- df %>%
  summarise(
    mean = mean(`#galleries`, na.rm = TRUE),
    variance = var(`#galleries`, na.rm = TRUE),
    min = min(`#galleries`, na.rm = TRUE),
    max = max(`#galleries`, na.rm = TRUE),
    zeros = sum(`#galleries` == 0, na.rm = TRUE)
  )

fluency_distribution


## 4.3 Overall histogram of fluency ----

p_fluency_hist_overall <- ggplot(
  df,
  aes(x = `#galleries`)
) +
  geom_histogram(
    bins = 20,
    fill = "grey75",
    color = "white"
  ) +
  labs(
    x = "Number of shapes saved to the gallery",
    y = "Number of participants"
  ) +
  theme_classic(base_size = 14)

p_fluency_hist_overall


## 4.4 Histogram of fluency by group ----

p_fluency_hist_group <- ggplot(
  df,
  aes(x = `#galleries`, fill = group)
) +
  geom_histogram(
    bins = 20,
    alpha = 0.45,
    position = "identity",
    color = "white"
  ) +
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  labs(
    x = "Number of shapes saved to the gallery",
    y = "Number of participants",
    fill = NULL
  ) +
  theme_classic(base_size = 14)

p_fluency_hist_group


## 4.5 Density plot of fluency by group ----

p_fluency_density <- ggplot(
  df,
  aes(
    x = `#galleries`,
    fill = group,
    color = group
  )
) +
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  labs(
    x = "Number of shapes saved to the gallery",
    y = "Density",
    fill = NULL,
    color = NULL
  ) +
  theme_classic(base_size = 14)

p_fluency_density


# 5. Create clean fluency variable ----

df <- df %>%
  mutate(
    fluency = `#galleries`
  )


# 6. Bayesian fluency model ----

## 6.1 Priors ----

priors_fluency <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(log(40), 0.5), class = "Intercept"),
  prior(exponential(0.5), class = "shape")
)


## 6.2 Negative binomial regression:
## ADHD vs Without ADHD ----

m_fluency <- brm(
  fluency ~ group_c,
  data = df,
  family = negbinomial(link = "log"),
  
  prior = priors_fluency,
  
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

summary(m_fluency)


# 7. Posterior summary ----

fluency_posterior <- describe_posterior(
  m_fluency,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

fluency_posterior


# 8. Posterior draws and back-transformation ----

fluency_draws <- as_draws_df(m_fluency) %>%
  mutate(
    
    # Estimated log-count for each group
    without_adhd_log =
      b_Intercept - 0.5 * b_group_c,
    
    adhd_log =
      b_Intercept + 0.5 * b_group_c,
    
    
    # Back-transform to expected number of saved shapes
    without_adhd_fluency =
      exp(without_adhd_log),
    
    adhd_fluency =
      exp(adhd_log),
    
    
    # ADHD - Without ADHD difference
    # in expected number of saved shapes
    diff_fluency =
      adhd_fluency - without_adhd_fluency,
    
    
    # Relative difference between groups
    rate_ratio =
      exp(b_group_c)
  )


# 8.1 Posterior estimates for each group ----

fluency_group_summary <- tibble(
  group = c(
    "Without ADHD",
    "ADHD"
  ),
  
  posterior_median = c(
    median(fluency_draws$without_adhd_fluency),
    median(fluency_draws$adhd_fluency)
  ),
  
  lower_90 = c(
    quantile(
      fluency_draws$without_adhd_fluency,
      0.05
    ),
    quantile(
      fluency_draws$adhd_fluency,
      0.05
    )
  ),
  
  upper_90 = c(
    quantile(
      fluency_draws$without_adhd_fluency,
      0.95
    ),
    quantile(
      fluency_draws$adhd_fluency,
      0.95
    )
  )
)

fluency_group_summary


# 8.2 ADHD - Without ADHD difference ----

fluency_difference <- describe_posterior(
  fluency_draws$diff_fluency,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

fluency_difference


# 8.3 Rate ratio ----

fluency_rate_ratio <- describe_posterior(
  fluency_draws$rate_ratio,
  centrality = "median",
  ci = 0.90
)

fluency_rate_ratio


# ============================================================
# FIGURES: MAIN ADHD vs WITHOUT ADHD COMPARISON
# ============================================================


# 9. Plot A: raw observed fluency by group ----

p_fluency_raw <- ggplot(
  df,
  aes(
    x = group,
    y = fluency,
    color = group
  )
) +
  
  # Individual participants
  geom_jitter(
    width = 0.12,
    alpha = 0.30,
    size = 2,
    show.legend = FALSE
  ) +
  
  # Mean ± 1 SD
  stat_summary(
    fun.data = mean_sdl,
    fun.args = list(mult = 1),
    geom = "errorbar",
    width = 0.05,
    linewidth = 0.6,
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  # Group mean
  stat_summary(
    fun = mean,
    geom = "point",
    size = 3.5,
    color = "black",
    show.legend = FALSE
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  
  labs(
    x = NULL,
    y = "Fluency (number of saved shapes)"
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    axis.text = element_text(
      size = 13
    ),
    
    axis.title = element_text(
      size = 15
    )
  )

p_fluency_raw


# 10. Plot B: posterior distributions for each group ----

fluency_groups_long <- fluency_draws %>%
  select(
    without_adhd_fluency,
    adhd_fluency
  ) %>%
  rename(
    `Without ADHD` = without_adhd_fluency,
    ADHD = adhd_fluency
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "fluency"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "ADHD",
        "Without ADHD"
      )
    )
  )


# Posterior median for each group

fluency_group_medians <- fluency_groups_long %>%
  group_by(group) %>%
  summarise(
    median = median(fluency),
    .groups = "drop"
  )


p_fluency_groups <- ggplot(
  fluency_groups_long,
  aes(
    x = fluency,
    fill = group,
    color = group
  )
) +
  
  # Posterior distributions
  geom_density(
    alpha = 0.65,
    linewidth = 0.8,
    adjust = 1
  ) +
  
  # Grey baseline
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  # Black points = posterior medians
  geom_point(
    data = fluency_group_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  labs(
    x = "Estimated fluency",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(
      size = 13
    ),
    
    axis.title.x = element_text(
      size = 15
    ),
    
    legend.position = "right",
    legend.text = element_text(
      size = 13
    ),
    
    aspect.ratio = 0.45
  )

p_fluency_groups


# 11. Plot C: posterior ADHD - Without ADHD difference ----

## 11.1 Posterior median ----

fluency_diff_median <- median(
  fluency_draws$diff_fluency,
  na.rm = TRUE
)


## 11.2 Difference plot ----

p_fluency_diff <- ggplot(
  fluency_draws,
  aes(x = diff_fluency)
) +
  
  # Posterior distribution
  geom_density(
    fill = "#808080",
    alpha = 0.40,
    color = "#555555",
    linewidth = 1
  ) +
  
  # Zero = no group difference
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  # Posterior median
  annotate(
    "point",
    x = fluency_diff_median,
    y = 0,
    size = 3,
    color = "black"
  ) +
  
  labs(
    x = "Estimated ADHD - Without ADHD difference in fluency",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(
      size = 13
    ),
    
    axis.title.x = element_text(
      size = 15
    ),
    
    legend.position = "none",
    
    aspect.ratio = 0.45
  )

p_fluency_diff


# 12. Save main fluency plots ----

dir.create(
  "Figures/fluency",
  recursive = TRUE,
  showWarnings = FALSE
)


ggsave(
  filename = "Figures/fluency/fluency_raw_by_group.png",
  plot = p_fluency_raw,
  width = 6,
  height = 5,
  dpi = 300,
  bg = "white"
)


ggsave(
  filename = "Figures/fluency/fluency_posterior_groups.png",
  plot = p_fluency_groups,
  width = 6,
  height = 4,
  dpi = 300,
  bg = "white"
)


ggsave(
  filename = "Figures/fluency/fluency_group_difference.png",
  plot = p_fluency_diff,
  width = 6,
  height = 5,
  dpi = 300,
  bg = "white"
)


# ============================================================
# ADHD PRESENTATION ANALYSIS: FLUENCY
# ============================================================


# 13. Define presentation groups ----

df <- df %>%
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
    ),
    
    fluency = `#galleries`
  )

table(
  df$presentation3,
  useNA = "ifany"
)


# 14. Priors ----

priors_fluency_presentation <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(log(40), 0.5), class = "Intercept")
)


# 15. Bayesian model: fluency across ADHD presentations ----

m_fluency_presentation <- brm(
  fluency ~ presentation3,
  
  data = df,
  family = negbinomial(link = "log"),
  
  prior = priors_fluency_presentation,
  
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

summary(m_fluency_presentation)


# 16. Posterior expected fluency for each presentation group ----

newdata_fluency_presentation <- tibble(
  presentation3 = factor(
    c(
      "Without ADHD",
      "Inattentive",
      "Combined/HI"
    ),
    levels = levels(df$presentation3)
  )
)


fluency_presentation_epred <- posterior_epred(
  m_fluency_presentation,
  newdata = newdata_fluency_presentation
)


# 17. Posterior summary by presentation group ----

fluency_presentation_summary <- tibble(
  group = c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  ),
  
  posterior_median = apply(
    fluency_presentation_epred,
    2,
    median
  ),
  
  lower_90 = apply(
    fluency_presentation_epred,
    2,
    quantile,
    probs = 0.05
  ),
  
  upper_90 = apply(
    fluency_presentation_epred,
    2,
    quantile,
    probs = 0.95
  )
)

fluency_presentation_summary


# 18. Pairwise posterior contrasts ----

fluency_presentation_contrasts <- tibble(
  
  contrast = c(
    "Inattentive - Without ADHD",
    "Combined/HI - Without ADHD",
    "Combined/HI - Inattentive"
  ),
  
  median = c(
    
    median(
      fluency_presentation_epred[, 2] -
        fluency_presentation_epred[, 1]
    ),
    
    median(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 1]
    ),
    
    median(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 2]
    )
  ),
  
  lower_90 = c(
    
    quantile(
      fluency_presentation_epred[, 2] -
        fluency_presentation_epred[, 1],
      0.05
    ),
    
    quantile(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 1],
      0.05
    ),
    
    quantile(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 2],
      0.05
    )
  ),
  
  upper_90 = c(
    
    quantile(
      fluency_presentation_epred[, 2] -
        fluency_presentation_epred[, 1],
      0.95
    ),
    
    quantile(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 1],
      0.95
    ),
    
    quantile(
      fluency_presentation_epred[, 3] -
        fluency_presentation_epred[, 2],
      0.95
    )
  )
)

fluency_presentation_contrasts


# 19. Plot: posterior fluency distributions
# across ADHD presentations ----

fluency_presentation_long <- as_tibble(
  fluency_presentation_epred
) %>%
  setNames(
    c(
      "Without ADHD",
      "Inattentive",
      "Combined/HI"
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "fluency"
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


fluency_presentation_medians <- fluency_presentation_long %>%
  group_by(group) %>%
  summarise(
    median = median(fluency),
    .groups = "drop"
  )


p_fluency_presentation <- ggplot(
  fluency_presentation_long,
  aes(
    x = fluency,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_point(
    data = fluency_presentation_medians,
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
  
  labs(
    x = "Estimated fluency (number of saved shapes)",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    panel.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.grid = element_blank(),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(
      size = 13
    ),
    
    axis.title.x = element_text(
      size = 15
    ),
    
    legend.position = "right",
    legend.text = element_text(
      size = 13
    ),
    
    aspect.ratio = 0.45
  )

p_fluency_presentation


# 20. Save fluency presentation plot ----

ggsave(
  filename = "Figures/fluency/fluency_posterior_by_presentation.png",
  plot = p_fluency_presentation,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)


# 21. Posterior contrast:
# Inattentive - Combined/HI ----

fluency_diff_inatt_vs_combined <- tibble(
  difference =
    fluency_presentation_epred[, 2] -
    fluency_presentation_epred[, 3]
)


fluency_diff_summary <- fluency_diff_inatt_vs_combined %>%
  summarise(
    median = median(difference),
    lower_90 = quantile(
      difference,
      0.05
    ),
    upper_90 = quantile(
      difference,
      0.95
    )
  )

fluency_diff_summary


# 22. Plot posterior contrast between ADHD presentations ----

p_fluency_inatt_vs_combined <- ggplot(
  fluency_diff_inatt_vs_combined,
  aes(x = difference)
) +
  
  geom_density(
    fill = "#0072B2",
    color = "#0072B2",
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = fluency_diff_summary,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  labs(
    x = "Estimated Inattentive - Combined/HI difference in fluency",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    panel.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.grid = element_blank(),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(
      size = 13
    ),
    
    axis.title.x = element_text(
      size = 15
    ),
    
    aspect.ratio = 0.45
  )

p_fluency_inatt_vs_combined


# 23. Save ADHD-presentation contrast plot ----

ggsave(
  filename =
    "Figures/fluency/fluency_inattentive_vs_combinedHI_difference.png",
  plot = p_fluency_inatt_vs_combined,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)


# 24. pd for Inattentive - Combined/HI contrast ----

fluency_diff_inatt_vs_combined_vector <-
  fluency_presentation_epred[, 2] -
  fluency_presentation_epred[, 3]


fluency_pd_inatt_vs_combined <-
  max(
    mean(
      fluency_diff_inatt_vs_combined_vector > 0
    ),
    mean(
      fluency_diff_inatt_vs_combined_vector < 0
    )
  ) * 100

fluency_pd_inatt_vs_combined