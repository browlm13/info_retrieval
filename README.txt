TODO: Postings lists and tiered postings lists - weighted scores
TODO: Postings list <-> Term Frequency Matrix
TODO: Impalement Kmeans clustering for document vector groupings
TODO: Purity Measures of clustering

----------------------------------------
Program use:
python __main__.py -o [output_directory_name (Required)] -n [max_number_to_index] -i [stopwords_file.txt] -u [seed_url]
-----------------------------------------
dependencies:
    nltk - natural language tool kit
    glob
    bs4  - beautiful soup
    pandas
-----------------------------------------
Information reported in output folder:
output/summary_log.txt
output/document_term_frequency_matrix.csv
----------------------------------------
Data Structures Used:
Queue for URL frontier
Hash tables for indexing Documents and Urls
Bots and CNC type architecture
*md5 for detecting duplicates
------------------------------------------
Output And Collected Data Directory Structure:
------------------------------------------
collected_data/
    resolved_url_map.json
    url_id_map.json
    doc_hash_id_map.json

    "NAMED_OUTPUT_DIRECTORY"/
        log.txt
        response_summaries/
            response_summary_0.json
            ...
        documents/
            document_frequency_dict_0.json
            ...
output/
    document_term_frequency_matrix.csv
    output_log.txt
    summary_log.txt
------------------------------------------
\Output And Collected Data Directory Structure
------------------------------------------
