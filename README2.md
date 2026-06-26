# 🏙️ City and Weather Assistant 

A beautiful, interactive Streamlit application for your LangChain city assistant. Get weather and news information about Indian cities with an elegant approval workflow.

## ✨ Features

- **Modern Streamlit UI**: Clean, responsive interface with custom styling
- **Real-time Chat**: Have conversations with the AI assistant
- **Smart Approval System**: Review and approve/deny tool calls before execution
- **Live Weather Data**: Get current weather for any Indian city
- **Latest News**: Fetch recent news articles about cities
- **Chat History**: Persistent conversation tracking with thread management
- **Settings Panel**: Manage thread ID and clear chat history
- **Mobile Friendly**: Responsive design that works on all devices

## 🎓 How It Works

1. **User Input**: You type a message in the chat box
2. **Agent Processing**: LangChain agent analyzes your request
3. **Tool Decision**: Agent decides if it needs to call weather/news API
4. **Approval Wait**: If tools needed, approval card appears
5. **User Approval**: You approve or deny the tool call
6. **Tool Execution**: If approved, the API is called
7. **Response**: Agent generates response with the data
8. **Display**: Message appears in chat history

## 🔗 API Integrations

### OpenWeather
- Gets current weather for any location in India
- Returns: temperature, description, humidity, wind speed
- Rate limit: 1000 calls/day (free tier)

### Tavily Search
- Searches for latest news about cities
- Returns: article titles, snippets, URLs
- Rate limit: Based on your plan

## 📝 License

Open source and available under MIT License.

---

**Enjoy your City Assistant! 🎉**

For updates and improvements, check the project repository regularly.
