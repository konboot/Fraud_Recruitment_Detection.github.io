<?php
   $Email = $_POST['Email'];
   $Subject = $_POST['Subject'];
   $Message = $_POST['Message'];

   $conn = new mysqli('127.0.0.1','root','','test');
   if($conn->connect_error){
       die('Connection Failed : '.$conn->connect_error);
   }else{
       $stmt = $conn->prepare("insert into registration(Email, Subject, Message)
       values(?,?,?)");
       $stmt->bind_param("sss",$Email,$Subject,$Message);
       $stmt->execute();
       echo "Registration Successfully...";
       $stmt->close();
       $conn->close();

   }
?>