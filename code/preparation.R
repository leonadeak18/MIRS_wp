#library for this code
library(prospectr)
library(mdatools)

# Prepare the dataframe
prepare_df <- function(input, r1, r2){
  samples <- input[,1]
  df <- input[,-1]
  colnames(df) <- sub("^X", "", colnames(df))
  cols <- as.numeric(colnames(df))
  output <- cbind(samples, df[, cols >= r1 & cols <= r2])
  
  return(output)
}

# Baseline Correction
baseline_corr <- function(input){
  df <- as.matrix(input[,-1])
  wav <- as.numeric(colnames(df))
  bs <- baseline(df, wav)
  bs <- as.data.frame(bs)
  output <- cbind(samples = input[,1],bs)
  
  return(output)
}

# Smoothing
smoothing <- function(input, width, porder){
  # Remove the first column
  df <- as.matrix(input[, -1])
  model <- prep.savgol(df, width = width, porder = porder, dorder = 0)
  model <- as.data.frame(model)
  output <- cbind(samples = input[,1], model)
  
  return(output)
}

# Derivative
derivative <- function(input, width, porder, dorder){
  df <- input[,-1]
  df <- as.matrix(df)
  sgvec <- savitzkyGolay(df, w = width, p = porder, m = dorder)
  output <- as.data.frame(cbind(samples = input[,1], sgvec))
  
  return (output)
}

#combining all spectra
combine <- function(input){
  pname <- c("alb.", "glo.", "gli.", "glu.")
  rename_col <- lapply(seq_along(input), function(i){
    df <- input[[i]][,-1]
    colnames(df) <- paste0(pname[i], colnames(df))
    return(df)
  })
  output <- do.call(cbind, rename_col)
}
