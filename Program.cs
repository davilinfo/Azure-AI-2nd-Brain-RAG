using Azure;
using Azure.AI.OpenAI;
using Azure.AI.TextAnalytics;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using Microsoft.Extensions.Configuration;
using OpenAI.Chat;
using System.IO;
using System.Text;

// See https://aka.ms/new-console-template for more information
namespace The2ndBrain{
    public class Program{
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
        static OpenAI.OpenAIClient _openAIClient;

        static SearchClient _searchClient;
        static TextAnalyticsClient _textAnalyticsClient;
        static IConfigurationRoot _configurationRoot;
        public static async Task Main(string[] args){
            IConfigurationBuilder builder = new ConfigurationBuilder().AddJsonFile("appsettings.json");
            _configurationRoot = builder.Build();                        

            await CreateIndex();
#pragma warning disable CS8604 // Possible null reference argument.
            _searchClient = new SearchClient(new Uri(_configurationRoot["AzureSearchIndexApi:endpoint"]), _configurationRoot["AzureSearchIndexApi:index"], new AzureKeyCredential(_configurationRoot["AzureSearchIndexApi:key"]));

            _openAIClient = new AzureOpenAIClient(new Uri(_configurationRoot["LLM:AzureOpenAI:endpoint"]), new System.ClientModel.ApiKeyCredential(_configurationRoot["LLM:AzureOpenAI:key"]));
            _textAnalyticsClient = new TextAnalyticsClient(new Uri(_configurationRoot["TextAnalytics:endpoint"]), new AzureKeyCredential(_configurationRoot["TextAnalytics:key"]));
            
            try{
                await GenerateTextFromOpenAI();
            }catch(Exception e){
                Console.WriteLine($"Exception in ", e.Message);
            }
        }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
        private static async Task GenerateFinalResolution(string topic){

            var systemPrompt = """                
                You are a critical assistant that will analize the following topic {0}.
                Answer the query using only the sources provided below in a critical and unique perspective.
                Answer ONLY with the facts listed in the list of sources below.
                Based on your research and insights, write a Spiky POV about any subtopic in 1-2 sentences. 
                Ensure it is distinct, thought-provoking, and applicable by being relevant and practical to the field. The Spiky POV should challenge traditional perspectives and provide a fresh take on the topic
                Do not generate answers that don't use the sources below.
                Query: {1}
                Sources:\n{2}
                """;

            var messages = new OpenAI.Chat.ChatMessage[] {
                    new SystemChatMessage("Your name is Jarvis"),                
                    new UserChatMessage($"Generate a brief of such topic {topic}."
                    +"\n Utilize the source" )
            };                
            
            var chatModel = "gpt-35-turbo";
            var response = _openAIClient.GetChatClient(chatModel).CompleteChat(
                messages: messages              
            );

            Console.WriteLine("-------------------");
            Console.WriteLine($"Brief of topic {topic} with 70 words maximum: {response.Value?.Content[0].Text} \n");
            
            var query = $"Can you retrieve a critical perspective about the topic {topic}?";

            var search_options = new SearchOptions();            
            search_options.SearchFields.Add("Topic");
            search_options.SearchFields.Add("Title");
            search_options.SearchFields.Add("Summary");            
            search_options.SearchFields.Add("Sentiment");
                        
            var search_results = _searchClient.Search<SearchDocument>(
                searchText: query, 
                options: search_options, 
                cancellationToken: default);            

            var source_formatted = new StringBuilder();
            source_formatted.AppendJoin("\n", 
                (from document in search_results.Value.GetResults()
                select $"Topic:{document.Document["Topic"]};Title:{document.Document["Title"]};Content:{document.Document["Summary"]};Sentiment:{document.Document["Sentiment"]};"));
            ;

            messages = new OpenAI.Chat.ChatMessage[] {
                    new SystemChatMessage("Your name is Jarvis"),                
                    new UserChatMessage(string.Format(systemPrompt, topic, query, source_formatted))
            };
            
            var finalResponse = _openAIClient.GetChatClient(chatModel).CompleteChat(
                messages: messages              
            );
            
            Console.WriteLine("-------------------");
            Console.WriteLine($"Final resolution about {topic}: {finalResponse.Value?.Content[0].Text}");
        }

        /// <summary>
        /// Retrieved info from different sources and add to index
        /// Finalize call GenerateResolution to combine content from gen AI, author and open AI
        /// Write the Brien and final resolution
        /// </summary>
        private static async Task GenerateTextFromOpenAI(){            
            var listDocuments = new List<The2ndBrainIndex>();
            listDocuments.AddRange(await GetDocuments("learning"));            

            await AddDocumentsToindex(listDocuments);  

            await GenerateFinalResolution("learning");            
        }        

        /// <summary>
        /// Utilize text analizer for verify the sentiment of the text
        /// </summary>
        /// <param name="topic"></param>
        /// <returns></returns>
        private static async Task<List<The2ndBrainIndex>> GetDocuments(string topic){
            var listDocuments = new List<The2ndBrainIndex>();
            var texts = TopicTextFromDifferentSources(topic);                        
            foreach(var textFromGenAI in texts){                                
                var link = await GetResponse($"Retrieve the Link as string of this text. If you cannot retrieve then return empty string: {textFromGenAI}");
                var author = await GetResponse($"Retrieve the Author of this text. If you cannot retrieve then return empty string: {textFromGenAI}");
                var content = await GetResponse($"Retrieve the Summary of this text. If you cannot retrieve then return empty string: {textFromGenAI}");                                
                var title = await GetResponse($"Retrieve the Title of this text. If you cannot retrieve then return empty string: {textFromGenAI}");
                var sentiment = await _textAnalyticsClient.AnalyzeSentimentAsync(textFromGenAI);                
                
                var document = new The2ndBrainIndex{
                    Id = Guid.NewGuid().ToString(),
                    Author = author ?? "gemini",
                    Topic = topic,
                    Summary = content,                    
                    Title = title,
                    Link = link,
                    Sentiment = $"{sentiment.Value.Sentiment}"
                };

                listDocuments.Add(document);                
            }

            return listDocuments;
        }

        private static async Task<string> GetResponse(string prompt){
            var messages = new OpenAI.Chat.ChatMessage[] {
                    new SystemChatMessage("Your name is Jarvis"),                
                    new UserChatMessage($"{prompt}" )
            };                
            
            var chatModel = "gpt-35-turbo";
            var response = _openAIClient.GetChatClient(chatModel).CompleteChat(
                messages: messages              
            );

#pragma warning disable CS8603 // Possible null reference return.
            return response.Value?.Content[0].Text;
#pragma warning restore CS8603 // Possible null reference return.
        }

        private static async Task CreateIndex(){
            Uri endpoint = new Uri(_configurationRoot["AzureSearchIndexApi:endpoint"]); 
            AzureKeyCredential credential = new AzureKeyCredential(_configurationRoot["AzureSearchIndexApi:key"]); 
            SearchIndexClient indexClient = new SearchIndexClient(endpoint, credential); 
            
            try{                
                var response = await indexClient.DeleteIndexAsync(_configurationRoot["AzureSearchIndexApi:index"]);
            }catch(Exception e){
                Console.WriteLine("Index does not exist", e);
            }
            
            SearchIndex index = new SearchIndex(_configurationRoot["AzureSearchIndexApi:index"]) 
            { Fields = {                  
                new SearchField("Id", SearchFieldDataType.String) { IsKey = true, IsFilterable = true }, 
                new SearchField("Topic", SearchFieldDataType.String) { IsSearchable = true }, 
                new SearchField("Title", SearchFieldDataType.String) { IsSearchable = true }, 
                new SearchField("Link", SearchFieldDataType.String) { IsSearchable = true },
                new SearchField("Author", SearchFieldDataType.String) { IsSearchable = true },                
                new SearchField("Summary", SearchFieldDataType.String) { IsSearchable = true },                
                new SearchField("Sentiment", SearchFieldDataType.String) { IsSearchable = true }
                } 
            }; 
            indexClient.CreateIndex(index); 
            Console.WriteLine("Index created successfully.");
        }

        private static async Task AddDocumentsToindex(List<The2ndBrainIndex> documents){
            await _searchClient.UploadDocumentsAsync<The2ndBrainIndex>(documents);
            Console.WriteLine("Documents uploaded successfully to index.");
        }       

        /// <summary>
        /// Get the text from different sources of GenAI, author and also generate from OpenAi
        /// </summary>
        /// <param name="topic">topic to prompt or file to retrieve the text</param>
        /// <returns></returns>
        private static List<string> TopicTextFromDifferentSources(string topic){
            var listCommands = new List<string>();
            
            listCommands.AddRange(TextFromGenAI(topic));

            listCommands.AddRange(TextFromAuthor(topic));                        

            return listCommands;
        }        

        private static List<string> TextFromGenAI(string topic){
            var listCommands = new List<string>();
            var folderPath = Path.GetFullPath("./source_documents/llm_text_extraction/results/");
            DirectoryInfo folder = new DirectoryInfo(folderPath);
            foreach (var file in folder.GetFiles($"*{topic}.txt"))
            {
                // Read the file contents
                Console.WriteLine("\n-------------\n" + file.Name);
                StreamReader sr = file.OpenText();
                var text = sr.ReadToEnd();
                sr.Close();
                //Console.WriteLine("\n" + text);
                listCommands.Add(text);
            }

            return listCommands;
        }        

        private static List<string> TextFromAuthor(string topic){
            var listCommands = new List<string>();
            var folderPath = Path.GetFullPath("./source_documents/author_insights/");
            DirectoryInfo folder = new DirectoryInfo(folderPath);
            foreach (var file in folder.GetFiles($"*{topic}.txt"))
            {
                // Read the file contents
                Console.WriteLine("\n-------------\n" + file.Name);
                StreamReader sr = file.OpenText();
                var text = sr.ReadToEnd();
                sr.Close();                
                listCommands.Add(text);
            }
            return listCommands;
        }
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
#pragma warning restore CS8604 // Possible null reference argument.
    }

    public class The2ndBrainIndex {
        public string Id {get;set;}
        public string Topic {get;set;}
        public string Title { get; set; }
        public string Summary { get; set; }
        public string Author { get; set; }
        public string Link { get; set; }         
        public string Sentiment {get;set;}
    }
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
}
