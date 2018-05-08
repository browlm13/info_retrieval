
Finding the best number of clusters

maximizing the second derivative of the variance as a function of the number of clusters (k)
    1. plot the average? (throw out bad data) varaiance for each value of k
    2. use polynomial interpolation? for a function that describes the varaince as a function of k
    3. find the maximum of that functions second derivative



TODO: Revise structure

    objective:
        design system for multiple crawlers

            ## Crawler

            have crawler send "entire response" only if:
                content hash not in database

            1. give crawler url
            2. have crawler report back
                1. url list(original url, redirection urls (url history), and resolved url)
                2. response status
                3. content hash (if available)
                *4. time accessed

                (this can be used to generate content_hash : url_list dictionary)

            3. have base station send back instruction (Boolean)
                1. if content hash not in database instruct crawler to send entire "response" to base station
                    1. send binary response content to base station
                    2. end crawler job
                2. if content hash in database
                    1. end crawler job


            ## binary response parser

            1. hash content
            2. extract links (normalized)
                1. urls
                2. img urls
            3.


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
