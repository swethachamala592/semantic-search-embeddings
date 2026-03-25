from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Data
sentences = [
"I love watching movies",
"Python is easy to learn",
"Artificial intelligence is the future",
"I like eating pizza",
"Machine learning is interesting",
"I enjoy reading books",
"Data science is a growing field",
"I love playing cricket",
"Deep learning is a part of AI",
"I like listening to music",
"Programming improves problem solving skills",
"I enjoy traveling to new places",
"Technology is evolving rapidly",
"I like watching web series",
"Learning new skills is important",
"I enjoy coding in Python",
"I like exploring new technologies",
"AI is transforming industries",
"I love spending time with family",
"I enjoy cooking food",
"I like learning about space",
"Cloud computing is powerful",
"I enjoy playing video games",
"I like solving puzzles",
"Cybersecurity is very important",
"I enjoy watching documentaries",
"I like studying mathematics",
"Big data is used in analytics",
"I enjoy outdoor activities",
"I like learning new languages",
"Blockchain technology is trending",
"I enjoy drawing and painting",
"I like working on projects",
"I enjoy listening to podcasts",
"I like exploring nature",
"Mobile apps are useful",
"I enjoy writing code",
"I like watching tutorials",
"Automation saves time",
"I enjoy learning online",
"I like teamwork and collaboration",
"Internet makes life easier",
"I enjoy watching sports",
"I like improving my skills",
"I enjoy problem solving",
"I like building applications",
"AI helps in decision making",
"I enjoy helping others",
"I like learning daily",
"I enjoy reading tech blogs",
"I like creative thinking",
"I enjoy debugging code",
"I like experimenting with ideas",
"I enjoy attending workshops",
"I like developing software",
"I enjoy analyzing data",
"I like logical thinking",
"I enjoy learning algorithms",
"I like practicing coding",
"I enjoy working with data",
"I like building websites",
"I enjoy learning frameworks",
"I like exploring AI tools",
"I enjoy doing research",
"I like contributing to projects",
"I enjoy improving performance",
"I like solving real world problems",
"I enjoy learning new concepts",
"I like creating new things",
"I enjoy working in teams",
"I like exploring innovations",
"I enjoy building models",
"I like understanding systems",
"I enjoy working on challenges",
"I like developing skills",
"I enjoy achieving goals",
"I like staying updated",
"I enjoy learning technologies",
"I like coding daily",
"I enjoy solving coding problems",
"I like learning machine learning",
"I enjoy building AI models",
"I like understanding data",
"I enjoy exploring datasets",
"I like building smart systems",
"I enjoy innovation and creativity",
"I like improving efficiency",
"I enjoy technical learning",
"I like working on AI projects",
"I enjoy solving technical problems",
"I like learning programming languages",
"I enjoy building solutions",
"I like exploring computer science",
"I enjoy hands on practice",
"I like applying knowledge",
"I enjoy learning continuously",
"I like gaining experience",
"I enjoy developing ideas",
"I like working on technology"
]

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# User input
query = input("Enter your search: ")

# Convert query to embedding
query_embedding = model.encode(query)

# Store all distances
results = []

# Compare embeddings
for i in range(len(embeddings)):
    current_embedding = embeddings[i]
    distance = 0

    for j in range(len(query_embedding)):
        distance += (query_embedding[j] - current_embedding[j]) ** 2

    results.append((distance, sentences[i]))

# Sort by smallest distance
results.sort()

# Show top 3 matches
print("\nTop 3 Matches:\n")
for i in range(3):
    print(f"{i+1}. {results[i][1]}  (distance: {results[i][0]:.4f})")