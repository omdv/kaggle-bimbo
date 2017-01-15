library(stringr)
library(readr)
library(tm)
library(SnowballC)

preprocess_products <- function(product_data) {
  product_names <- as.character(product_data$NombreProducto)
  weight <- unlist(lapply(product_names, extract_weight))
  #volume <- unlist(lapply(product_names, extract_volume))
  pieces <- unlist(lapply(product_names, extract_pieces))
  brand <- unlist(lapply(product_names, extract_brand))
  product_shortname <- unlist(lapply(product_names, extract_shortname))
  has_choco <- unlist(lapply(product_names, grepl, pattern="Choco"))
  has_vanilla <- unlist(lapply(product_names, grepl, pattern="Va(i)?nilla"))
  has_multigrain <- unlist(lapply(product_names, grepl, pattern="Multigrano"))
  has_promotion <- unlist(lapply(product_names, grepl, pattern="Prom"))
  has_integral <- unlist(lapply(product_names, grepl, pattern="Int"))

  
  data.frame(
    ProductId=product_data$Producto_ID,
    product_name=product_names,
    product_shortname=product_shortname,
    product_brand=brand,
    product_weight=weight,
    product_pieces=pieces,
    #product_volume=volume,
    #product_wpp=weight/pieces,
    product_has_choco=has_choco,
    product_has_vanilla=has_vanilla,
    product_has_multigrain=has_multigrain,
    product_has_promotion=has_promotion,
    product_has_integral=has_integral
  )
}

extract_token <- function(value, expr) {
  tokens <- strsplit(value, " ")
  index <- grep(expr, tokens)
  ifelse(length(index) == 0, NA, tokens[index[1]])
}

extract_weight <- function(product_name) {
  weight_str <- extract_token(product_name, "\\d+[Kg|g|ml|kg|G]")
  if (is.na(weight_str)) return(NA)
  groups <- str_match_all(weight_str, "(\\d+)(Kg|g|ml|kg|G)")
  weight <- strtoi(groups[[1]][2])
  unit <- groups[[1]][3]
  ifelse(unit == "Kg" || unit == "kg", 1000 * weight, weight)
}

extract_volume <- function(product_name) {
  volume_str <- extract_token(product_name, "\\d+[ml]")
  if (is.na(volume_str)) return(NA)
  groups <- str_match_all(volume_str, "(\\d+)(ml)")
  return(strtoi(groups[[1]][2]))
}

extract_pieces <- function(product_name) {
  pieces_str <- extract_token(product_name, "\\d+p\\b")
  if (is.na(pieces_str)) return(NA)
  groups <- str_match_all(pieces_str, "(\\d+)(p)")
  return(strtoi(groups[[1]][2]))
}

extract_type <- function(product_name) {
  tokens <- strsplit(product_name, " ")[[1]]
  tokens[1]
}

extract_brand <- function(product_name) {
  tokens <- strsplit(product_name, " ")[[1]]
  tokens[length(tokens) - 1]
}

extract_shortname <- function(product_name) {
  # Split the name
  tokens <- strsplit(product_name, " ")[[1]]
  # Delete ID
  tokens <- head(tokens, length(tokens) - 1)
  # Delete Brands (name till the last token with digit)
  digit_indeces <- grep("[0-9]", tokens)
  # Product names without digits
  digit_index <- ifelse(length(digit_indeces) == 0,1,max(digit_indeces))
  paste(tokens[1:digit_index], collapse = " ")
}

extract_perks <- function(product_name) {
  # Split the name
  tokens <- strsplit(product_name, " ")[[1]]
  # Delete ID
  tokens <- head(tokens, length(tokens) - 1)
  # Delete Brands (name till the last token with digit)
  digit_indeces <- grep("[0-9]", tokens)
  # Product names without digits
  digit_index <- ifelse(length(digit_indeces) == 0,1,max(digit_indeces))
  paste(tokens[digit_index:length(tokens)-2], collapse = " ")
}

product_data <- read.csv('input/producto_tabla.csv')
product_data <- product_data[1:nrow(product_data),]
product_data <- preprocess_products(product_data)


# # ------------------------------------------
# # Short Names Processing
# CorpusShort <- Corpus(VectorSource(product_data$product_shortname))
# CorpusShort <- tm_map(CorpusShort, tolower)
# CorpusShort <- tm_map(CorpusShort, PlainTextDocument)

# # Remove Punctuation
# CorpusShort <- tm_map(CorpusShort, removePunctuation)

# # Remove Stopwords
# CorpusShort <- tm_map(CorpusShort, removeWords, stopwords("es"))

# # Stemming
# CorpusShort <- tm_map(CorpusShort, stemDocument, language="es")

# # Create DTM
# CorpusShort <- Corpus(VectorSource(CorpusShort))
# dtmShort <- DocumentTermMatrix(CorpusShort)

# # Delete Sparse Terms (all the words now)
# sparseShort <- removeSparseTerms(dtmShort, 0.9999)
# ShortWords <- as.data.frame(as.matrix(sparseShort))

# # Create valid names
# colnames(ShortWords) <- make.names(colnames(ShortWords))

# # Spherical k-means for product clustering (30 clusters at the moment)
# set.seed(123)
# mod <- skmeans(as.matrix(ShortWords), 30, method = "genetic")
# product_data$cluster <- mod$cluster
# # ------------------------------------------



write.csv(product_data, 'processed/processed_products.csv')
