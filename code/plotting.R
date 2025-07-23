#call all library needed
library(mdatools)
library(ggplot2)
library(viridis)
library(showtext)
library(ggpubr)
library(reshape2)
library(tidyr)

#load font
font_add("Arial", regular = "arial.ttf", bold = "arialbd.ttf")
showtext_auto()


calibration_curve <- function(input, x, y, col, shape){
  plot <- ggscatter(input, x = x, y = y, color = col, size = 3, shape = shape, 
                    add = "reg.line", add.params = list(linetype = "dotted")) +
    theme_classic(base_family = "Arial") +
    labs(y = "Integrated Area of Amide II (a.u.)", x = "Concentration (mg/mL)", shape = NULL, color = NULL) +
    scale_shape_manual(values = c(15:18)) +
    scale_color_manual(values = c("black", "black", "black", "black")) +
    geom_errorbar(aes(
      ymin = mean_auc - 2*sd_auc,
      ymax = mean_auc + 2*sd_auc,
      color = proteins), width = 0.1
    ) +
    theme(legend.position = "top",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 3)),
          panel.grid = element_blank(),
          axis.title = element_text(size = 12),
          axis.title.x = element_text(margin = margin(t = 3)),
          axis.text = element_text(size = 12),
          axis.title.y = element_text(margin = margin(r = 5)))
  return(plot)
}

box_plot <- function(df, x, y, col, title){
  ggplot(df, aes_string(x = x, y = y, color = col)) +
    geom_boxplot(aes_string(group = col)) +
    geom_jitter(aes_string(color = col), position = position_jitter(width = 0.2), alpha = 0.5, size =2) +
    scale_color_viridis_d(option = "D", direction = -1) +
    theme_classic() +
    labs(x = "Area", y = "Proteins (g / 100 g)", title = title, color = NULL) +
    theme(plot.title = element_text(hjust = 0.5))
}

comp_box_plot <- function(input, x, y, col){
  ggboxplot(input,
            x = x,
            y = y,
            color = col,
            palette = c("#fde725", "#35b779", "#31688e", "#440154")) +
    labs(y = "Proteins (g / 100 g)", x = "Protein Fractions", shape = NULL, color = NULL) +
    scale_x_discrete(labels = tools::toTitleCase) +
    ylim(0,7.5) +
    theme(legend.text = element_text(size = 12))
}

gbar_plot <- function(input, x, y, col, fill){
  plot <- ggbarplot(input, x = x, y = y, fill = fill, color = col) +
    scale_fill_manual(values = c("#e66101", "#fdb863", "#b2abd2", "#5e3c99")) +
    scale_color_manual(values = c("#e66101", "#fdb863", "#b2abd2", "#5e3c99")) +
    theme_bw() +
    labs(y = "Relative Protein Composition (%)", x = "Sampling Sites", fill = NULL, color = NULL) +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 12),
      plot.margin = margin(t = 15, r = 10, b = 10, l = 10),
      legend.key.width = unit(0.8, "cm"),
      axis.text = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.title.x = element_text(size = 12)
    )
  
  return(plot)
}

spectra_plot <- function(input, x, y, group, color, title){
  df_chart <- ggplot(input, aes(x = x, y = y, group = group, color=color)) +
    geom_line(linewidth = 0.75) +
    labs(x= expression("Wavenumber (cm"^{-1}*")"), y="Absorbance (a.u.)", title = title, color = NULL) +
    scale_color_viridis_d(option = "D", direction = -1) +
    theme_minimal(base_family = "Arial") +
    scale_x_reverse() +  # This will reverse the x-axis
    theme_bw() +
    theme(legend.position = "right",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 3)),
          panel.grid = element_blank(),
          axis.title = element_text(size = 12))
  # axis.title.x = element_text(margin = margin(t = 3)),
  # axis.title.y = element_text(margin = margin(r = 5)))
  
  return(df_chart)
}

spectra_plot2 <- function(input, x, y, group, color, title, ycap = ycap){
  df_chart <- ggplot(input, aes(x = x, y = y, group = group, color=color)) +
    geom_line(linewidth = 0.75) +
    labs(x= expression("Wavenumber (cm"^{-1}*")"), y=ycap, title = title, color = NULL) +
    scale_color_viridis_d(option = "D", direction = -1) +
    theme_minimal(base_family = "Arial") +
    scale_x_reverse() +  # This will reverse the x-axis
    theme_bw() +
    theme(legend.position = "right",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 3)),
          panel.grid = element_blank(),
          axis.title = element_text(size = 12))
  # axis.title.x = element_text(margin = margin(t = 3)),
  # axis.title.y = element_text(margin = margin(r = 5)))
  
  return(df_chart)
}

pca_plot <- function(input, input2, title) {
  # Calculate the farthest distance from the origin
  max_x <- 1.25 *max(abs(input$`Comp 1`))
  max_y <- 1.25 * max(abs(input$`Comp 2`))
  
  p <- ggplot(data = input, aes(x = `Comp 1`, y = `Comp 2`, color = location, shape = location)) +
    geom_point(size = 3) +
    scale_color_viridis_d(option = "D", direction = -1) +
    scale_shape_manual(values = c(15:18)) +
    theme_minimal(base_family = "Arial") +
    xlab(paste("PC1 (", round(input2$calres$expvar[1], 2), "%)", sep = "")) +
    ylab(paste("PC2 (", round(input2$calres$expvar[2], 2), "%)", sep = "")) +
    labs(color = NULL, title = title, shape = NULL) +
    xlim(-max_x, max_x) + 
    ylim(-max_y, max_y) + 
    theme_bw() +
    theme(legend.position = "top",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 1), size = 12, face = "bold"),
          panel.grid = element_blank(),
          axis.text = element_text(size = 12),
          axis.title = element_text(size = 12),
          axis.title.x = element_text(margin = margin(t = 1)),
          axis.title.y = element_text(margin = margin(r = 1))) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black") + # Dotted horizontal line at y=0
    geom_vline(xintercept = 0, linetype = "dotted", color = "black")
  
  
  return(p)
}

#PCA loading plot for the combined spectra
pca_combined_loading <- function(input, title){
  #create the dataframe first
  loadings <- input$loadings[, 1:2]
  xload <- paste("PC1 (", round(input$calres$expvar[1], 2), "%)")
  yload <- paste("PC2 (", round(input$calres$expvar[2], 2), "%)")
  colnames(loadings) <- c(xload, yload)
  
  #create a long version
  loadings_long <- melt(loadings)
  library(reshape2)
  loadings_long <- loadings_long %>% 
    separate(Var1, into = c("proteins", "wavenumber"), sep = "\\.", extra ="merge") %>%
    mutate(
      proteins = recode(proteins,
                        alb = "Albumins",
                        glo = "Globulins",
                        gli = "Gliadins",
                        glu = "Glutenins"),
      proteins = factor(proteins, levels = c("Albumins", "Globulins", "Gliadins", "Glutenins"))
    )
  loadings_long$wavenumber <- as.numeric(loadings_long$wavenumber)
  
  output <- ggplot(loadings_long, aes(x = wavenumber, y = value, color = Var2, group = Var2)) +
    geom_line() +
    scale_color_manual(values = c("#0072B2", "#E69F00")) +
    facet_wrap(~ proteins, nrow = 1) + # Separate charts by 'proteins'
    labs(x= expression("Wavenumber (cm"^{-1}*")"), y="PC Loading", title = title, color = NULL) +
    theme_minimal(base_family = "Arial") +
    scale_x_reverse() +
    theme_bw() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      legend.position = "top",
      legend.text = element_text(size = 12),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 12),
      panel.grid = element_blank(),
      legend.margin = margin(t = 0, b = -10),
      plot.title = element_text(hjust = 0.5, margin = margin(b = 1), size = 12, face = "bold"),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )
  
  return(output)
}

structure_plot <- function(input1, input2, col, label){
  max_points <- input1 %>%
    group_by(variable) %>%
    filter(value == max(value)) %>%
    slice(1) %>%
    ungroup()
  max_y <- max(max_points$value, na.rm = TRUE)
  if (max_y > max(input2[,3])){
    y_limit <- max_y*1.20
  }
  else{
    y_limit <- max(input2[,3])*1.20
  }
  
  p <- ggplot() +
    geom_line(data = input1, aes(x = wavenumber, y = value, color = variable), linetype = "dotted", size = 0.7) +
    geom_line(data = input2, aes(x = wavenumber, y = value), color = col, size = 1) +
    geom_text(data = max_points, aes(x = wavenumber, y = value, label = label), 
              color = "black", hjust = -0.1, size = 4, angle = 90) +
    labs(x= expression("Wavenumber (cm"^{-1}*")"), 
         y=expression("-d"^{2}*"A/dv"^{2}), color = NULL) +
    theme_minimal(base_family = "Arial") +
    scale_x_reverse() +
    scale_y_continuous(limits = c(NA, y_limit), labels = scales::scientific_format()) +
    theme_bw() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, margin = margin(b = 3)),
          panel.grid = element_blank(),
          axis.title = element_text(size = 12))
  return(p)
}

asca_plot <- function(input, ftr, title) {
  df <- as.data.frame(input$projected[[ftr]])
  df <- cbind(df, location = input$effects[[ftr]])
  # Calculate the farthest distance from the origin
  max_x <- 1.25 * max(abs(df$`Comp 1`))
  max_y <- 1.25 * max(abs(df$`Comp 2`))
  
  #calculate %VarianceExplained
  comp1 <- round(as.numeric(attr(scores(input, factor = ftr), "explvar")["Comp 1"]),1)
  comp2 <- round(as.numeric(attr(scores(input, factor = ftr), "explvar")["Comp 2"]),1)
  
  p <- ggplot(data = df, aes(x = `Comp 1`, y = `Comp 2`, color = location, shape = location)) +
    geom_point(size = 3) +
    scale_color_viridis_d(option = "D", direction = -1) +
    scale_shape_manual(values = c(15:18)) +
    theme_minimal(base_family = "Arial") +
    xlab(paste("PC1 (", round(comp1, 2), "%)", sep = "")) +
    ylab(paste("PC2 (", round(comp2, 2), "%)", sep = "")) +
    labs(color = NULL, title = title, shape = NULL) +
    xlim(-max_x, max_x) + 
    ylim(-max_y, max_y) + 
    theme_bw() +
    theme(legend.position = "top",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 1), size = 12, face = "bold"),
          panel.grid = element_blank(),
          axis.text = element_text(size = 12),
          axis.title = element_text(size = 12),
          axis.title.x = element_text(margin = margin(t = 1)),
          axis.title.y = element_text(margin = margin(r = 1))) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black") + # Dotted horizontal line at y=0
    geom_vline(xintercept = 0, linetype = "dotted", color = "black")
  
  
  return(p)
}

#Loading plot for ASCA
asca_loading <- function(input, ftr, title){
  #create the dataframe first
  df <- as.matrix(cbind(input$loadings[[ftr]]))
  loadings <- df[,c(1:2)]
  
  #calculate %VarianceExplained
  comp1 <- round(as.numeric(attr(scores(input, factor = ftr), "explvar")["Comp 1"]),1)
  comp2 <- round(as.numeric(attr(scores(input, factor = ftr), "explvar")["Comp 2"]),1)
  
  xload <- paste("PC1 (", comp1, "%)")
  yload <- paste("PC2 (", comp2, "%)")
  colnames(loadings) <- c(xload, yload)
  
  #create a long version
  loadings_long <- melt(loadings)
  library(reshape2)
  loadings_long <- loadings_long %>% 
    separate(Var1, into = c("proteins", "wavenumber"), sep = "\\.", extra ="merge") %>%
    mutate(
      proteins = recode(proteins,
                        alb = "Albumins",
                        glo = "Globulins",
                        gli = "Gliadins",
                        glu = "Glutenins"),
      proteins = factor(proteins, levels = c("Albumins", "Globulins", "Gliadins", "Glutenins"))
    )
  loadings_long$wavenumber <- as.numeric(loadings_long$wavenumber)
  
  output <- ggplot(loadings_long, aes(x = wavenumber, y = value, color = Var2, group = Var2)) +
    geom_line() +
    scale_color_manual(values = c("#0072B2", "#E69F00")) +
    facet_wrap(~ proteins, nrow = 1) + # Separate charts by 'proteins'
    labs(x= expression("Wavenumber (cm"^{-1}*")"), y="PC Loading", title = title, color = NULL) +
    theme_minimal(base_family = "Arial") +
    scale_x_reverse() +
    theme_bw() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      legend.position = "top",
      legend.text = element_text(size = 12),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 12),
      panel.grid = element_blank(),
      legend.margin = margin(t = 0, b = -10),
      plot.title = element_text(hjust = 0.5, margin = margin(b = 1), size = 12, face = "bold"),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )
  
  return(output)
}