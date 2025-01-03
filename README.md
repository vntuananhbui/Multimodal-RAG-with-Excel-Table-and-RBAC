
# AI Chatbot Project

An AI chatbot application built with **Streamlit**, A small project uses Multimodal RAG to extract employee information from a list in a PDF file and uses Chatbot via Streamlit to ask and answer questions about that PDF file. There are 3 main roles: "Employee", "Manager", "Admin", each position will have the right to extract different information based on decentralized data.

---

## **Features**
- **Pass Query Handling**: Normal information response*
- **Restricted Query Handling**: Only return "I'm sorry, but I can't help with that" *
---

## **Setup and Installation**

To run this project, follow these steps:

1. **Clone the Repository**  
   Clone this repository to your local system:

   ```bash
   git clone <https://github.com/vntuananhbui/Multimodal-RAG-with-Table-and-RBAC>
   cd <project-folder>
   ```

2. **Install Dependencies**  
   Install the required Python libraries using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Start the chatbot application with Streamlit:

   ```bash
   streamlit run App.py
   ```

4. **Access the Application**  
   Open the provided URL in your browser to start interacting with the chatbot.

---

## **How to Use**

1. **Launch the App**: Follow the setup instructions above to run the application.
2. **Upload PDF File**: Click on Browse files button on the sidebar and upload the pdf file in the repo data/data_employee.pdf.
3. **Select Role**: Prompt the query to find the information of the PDF file.

---

## **Technologies Used**

- **Python**
- **Streamlit**
- **LLamaIndex**
- **LLamaParse**
- **Presidio**


## **License**

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the license terms.

---

## **Contact**

For any questions, feedback, or suggestions, feel free to contact:

- **Email**: [vntuananhbui@gmail.com]
- **GitHub**: [https://github.com/vntuananhbui]
- **LinkedIn**: [https://www.linkedin.com/in/vntuananh/]

---

Happy coding! 🚀
