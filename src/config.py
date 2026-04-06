from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DEFAULT_WEIGHTS = {
    "tfidf": 0.20,
    "semantic": 0.30,
    "skill_match": 0.25,
    "hire_prob": 0.25,
}

RETRAIN_SAMPLE_THRESHOLD = 50
RETRAIN_AUC_THRESHOLD = 0.72

KAGGLE_DATASET = "gauravduttakiit/resume-dataset"

JOB_DESCRIPTIONS = {
    "Data Science": "Looking for a Data Scientist with Python, machine learning, SQL, pandas, numpy, scikit-learn, TensorFlow, PyTorch, data visualization, and statistics expertise.",
    "HR": "HR Manager needed with talent acquisition, employee relations, HRIS, payroll, onboarding, compliance, and performance management skills.",
    "Advocate": "Legal Advocate with civil/criminal litigation, contract drafting, legal research, court proceedings, IPC, civil law, and constitutional law knowledge.",
    "Arts": "Creative professional skilled in graphic design, illustration, Adobe Creative Suite, Photoshop, Illustrator, photography, and digital media.",
    "Web Designing": "Web Designer expert in HTML, CSS, JavaScript, responsive design, UI/UX, Figma, Adobe XD, WordPress, and Bootstrap.",
    "Mechanical Engineer": "Mechanical Engineer with AutoCAD, SolidWorks, ANSYS, thermodynamics, fluid mechanics, manufacturing processes, and CAD/CAM skills.",
    "Sales": "Sales Executive with B2B/B2C sales, CRM, lead generation, negotiation, pipeline management, Salesforce, and target-driven mindset.",
    "Health and fitness": "Fitness trainer with personal training certification, nutrition counseling, wellness coaching, exercise physiology, and gym management experience.",
    "Civil Engineer": "Civil Engineer with AutoCAD, STAAD Pro, structural design, project management, construction management, geotechnical, and IS codes knowledge.",
    "Java Developer": "Java Developer with Core Java, Spring Boot, Hibernate, microservices, RESTful APIs, Maven, SQL, JUnit, Docker, and AWS.",
    "Business Analyst": "Business Analyst with requirements gathering, JIRA, SQL, process modeling, stakeholder management, Agile, Scrum, and data analysis skills.",
    "SAP Developer": "SAP Consultant with ABAP, SAP FICO, SAP MM, SAP SD, SAP HR, S/4HANA, workflow, and SAP Basis expertise.",
    "Automation Testing": "QA Automation Engineer with Selenium, TestNG, JUnit, JMeter, API testing, Jenkins, CI/CD, Python, and Java skills.",
    "Electrical Engineering": "Electrical Engineer with AutoCAD Electrical, PLC, SCADA, power systems, circuit design, MATLAB, instrumentation, and IEC standards.",
    "Operations Manager": "Operations Manager with supply chain, lean manufacturing, Six Sigma, ERP, KPI tracking, process optimization, and logistics expertise.",
    "Python Developer": "Python Developer with Django, Flask, FastAPI, PostgreSQL, Redis, Celery, Docker, AWS, REST API, and machine learning pipeline skills.",
    "DevOps Engineer": "DevOps Engineer with Docker, Kubernetes, Jenkins, CI/CD, Terraform, AWS, Linux, Ansible, Prometheus, and Grafana expertise.",
    "Network Security Engineer": "Network Security Engineer with firewall, VPN, IDS/IPS, SIEM, penetration testing, Cisco, CISSP, CEH, and CCNA Security.",
    "PMO": "PMO professional with PMP, MS Project, JIRA, risk management, portfolio management, Agile, Waterfall, and governance frameworks.",
    "Database": "DBA with SQL, Oracle, MySQL, PostgreSQL, SQL Server, MongoDB, query optimization, backup/recovery, replication, and performance tuning.",
    "Hadoop": "Big Data Engineer with Hadoop, HDFS, MapReduce, Hive, Pig, Spark, Kafka, Scala, Python, and AWS EMR expertise.",
    "ETL Developer": "ETL Developer with Informatica, Talend, SSIS, SQL, data warehousing, dimensional modeling, Snowflake, and Redshift skills.",
    "DotNet Developer": ".NET Developer with C#, ASP.NET Core, Entity Framework, SQL Server, REST API, MVC, Azure, and microservices.",
    "Blockchain": "Blockchain Developer with Solidity, Ethereum, smart contracts, Web3.js, Truffle, IPFS, DeFi, and cryptography expertise.",
    "Testing": "QA Engineer with manual testing, test cases, JIRA, regression testing, bug reporting, Selenium, API testing, and quality assurance.",
}

CATEGORY_SKILLS = {
    "Data Science": ["python", "machine learning", "sql", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "statistics", "data visualization"],
    "HR": ["recruitment", "performance management", "hris", "payroll", "employee relations", "talent acquisition", "onboarding", "compliance"],
    "Advocate": ["litigation", "legal research", "contract drafting", "court proceedings", "ipc", "civil law", "criminal law", "constitutional law"],
    "Arts": ["photoshop", "illustrator", "graphic design", "animation", "photography", "after effects", "indesign", "creative suite"],
    "Web Designing": ["html", "css", "javascript", "figma", "adobe xd", "ui/ux", "responsive design", "wordpress", "bootstrap"],
    "Mechanical Engineer": ["autocad", "solidworks", "ansys", "catia", "cad", "manufacturing", "thermodynamics", "fluid mechanics", "matlab"],
    "Sales": ["crm", "salesforce", "lead generation", "b2b", "b2c", "negotiation", "pipeline management", "account management"],
    "Health and fitness": ["personal training", "nutrition", "wellness", "exercise physiology", "first aid", "cpr", "fitness assessment"],
    "Civil Engineer": ["autocad", "staad pro", "revit", "structural design", "construction management", "geotechnical", "is codes"],
    "Java Developer": ["java", "spring boot", "hibernate", "maven", "microservices", "rest api", "sql", "junit", "docker", "aws"],
    "Business Analyst": ["requirements gathering", "jira", "sql", "process modeling", "stakeholder management", "agile", "scrum", "data analysis"],
    "SAP Developer": ["sap", "abap", "sap fico", "sap mm", "sap sd", "sap hr", "s/4hana", "workflow", "sap basis"],
    "Automation Testing": ["selenium", "testng", "junit", "jmeter", "api testing", "jenkins", "ci/cd", "python", "java"],
    "Electrical Engineering": ["autocad electrical", "plc", "scada", "power systems", "circuit design", "matlab", "instrumentation"],
    "Operations Manager": ["supply chain", "lean manufacturing", "six sigma", "erp", "kpi", "process optimization", "logistics"],
    "Python Developer": ["python", "django", "flask", "fastapi", "postgresql", "redis", "celery", "docker", "aws", "rest api"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "aws", "linux", "ansible", "prometheus", "grafana"],
    "Network Security Engineer": ["firewall", "vpn", "ids/ips", "siem", "penetration testing", "cisco", "cissp", "ceh", "ccna"],
    "PMO": ["pmp", "ms project", "jira", "risk management", "portfolio management", "agile", "waterfall", "governance"],
    "Database": ["sql", "oracle", "mysql", "postgresql", "sql server", "mongodb", "query optimization", "backup recovery", "pl/sql"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce", "hive", "pig", "spark", "kafka", "scala", "python", "aws emr"],
    "ETL Developer": ["informatica", "talend", "ssis", "sql", "data warehousing", "etl", "snowflake", "redshift", "data modeling"],
    "DotNet Developer": ["c#", "asp.net", ".net core", "entity framework", "sql server", "rest api", "mvc", "azure", "microservices"],
    "Blockchain": ["solidity", "ethereum", "smart contracts", "web3.js", "truffle", "ipfs", "defi", "cryptography"],
    "Testing": ["manual testing", "test cases", "jira", "regression testing", "bug reporting", "selenium", "api testing", "quality assurance"],
}
