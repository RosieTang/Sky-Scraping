import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [uploadedPhoto, setUploadedPhoto] = useState(null);
  const [returnedPhoto, setReturnedPhoto] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const handlePhotoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedPhoto(file);
    }
  };
  

  useEffect(() => {
    // Simulate a loading delay (e.g., waiting for data to load)
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000); // After 2 seconds, the loading screen will disappear. Adjust this value as needed.

    return () => clearTimeout(timer); // Cleanup the timeout if the component is unmounted before the timeout finishes
  }, []);

  const handleSubmit = async () => {
    // if (!uploadedPhoto) return;

    // const formData = new FormData();
    // formData.append('photo', uploadedPhoto);

    // try {
    //     const response = await fetch('http://localhost:5173/upload', {
    //         method: 'POST',
    //         body: formData
    //     });

    //     if (response.ok) {
    //         const data = await response.json();
            
    //         // If using base64 encoding:
    //         setReturnedPhoto("data:image/jpeg;base64," + data.image);
            
    //         console.log('Photo uploaded and processed successfully');
    //     } else {
    //         console.error('Error uploading photo');
    //     }
    // } catch (error) {
    //     console.error('There was an error uploading the photo', error);
    // }
  };


  
  useEffect(() => {
    document.title = "Sky Scraping";
  }, []);

  return (
    <>
      {isLoading && (
        <div className="loading-overlay">
          <div className="curtain left"></div>
          <div className="curtain right"></div>
          <div className="ball red"></div>
          <div className="ball yellow"></div>
        </div>
      )}


      <div className="nav">
        Sky Scraping
      </div>
      <div className="photo">
        <h1 className="centered-header">Become a constellation!</h1>
        <div className="upload-section">
          {uploadedPhoto ? (
            <img src={URL.createObjectURL(uploadedPhoto)} alt="Uploaded preview" className="uploaded-preview" />
            ) : (
              <label className="upload-label">
              Drop your photo here or click to select one
              <input type="file" onChange={handlePhotoUpload} accept=".jpg" style={{ display: 'none' }} />
            </label>
          )}
        </div>
        <button onClick={handleSubmit}>Submit</button>
        {returnedPhoto && (
          <div className="returned-section">
            <h2>Returned Picture</h2>
            <img src={returnedPhoto} alt="Returned content" className="returned-preview" />
          </div>
        )}
      </div>
      
      <div className="THE PROJECT">
        <h1>About Our Project</h1>
      </div>
      <div className="ABOUT US">
        <h1>About us</h1>
      </div>
      
    </>
  );
}

export default App;
