import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App';
import './theme.css';

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0A84FF',
    },
    background: {
      default: '#F0F2F5',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1C1C1E',
    },
  },
});

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <ThemeProvider theme={lightTheme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
); 