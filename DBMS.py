from tkinter import *
from tkinter import ttk
import datetime
import time
import tkinter.messagebox
import sqlite3


class Hospital_Portal:
    db_name = 'Database.db'

    def __init__(self, root):
        self.root = root
        self.root.geometry('800x550+600+200')
        self.root.title('User Data')

        self.photo = PhotoImage(file='GUIimage\icon.png')
        self.label = Label(image=self.photo)
        self.label.grid(row=0, column=0)

        self.label1 = Label(font=('arial', 15, 'bold'),
                            text='Hospital Portal System', fg='dark blue')
        self.label1.grid(row=8, column=0)

        frame = LabelFrame(self.root, text='Add New Record:')
        frame.grid(row=0, column=1)

        Label(frame, text='ID Number:').grid(row=1, column=1, sticky=W)
        self.IDnumber = Entry(frame)
        self.IDnumber.grid(row=1, column=2)

        Label(frame, text='Password:').grid(row=2, column=1, sticky=W)
        self.Password = Entry(frame)
        self.Password.grid(row=2, column=2)

        Label(frame, text='Date Of Birth:').grid(row=3, column=1, sticky=W)
        self.Birth = Entry(frame)
        self.Birth.grid(row=3, column=2)

        Label(frame, text='Event:').grid(row=4, column=1, sticky=W)
        self.Event = Entry(frame)
        self.Event.grid(row=4, column=2)

        Label(frame, text='Medic:').grid(row=5, column=1, sticky=W)
        self.Medic = Entry(frame)
        self.Medic.grid(row=5, column=2)

        ttk.Button(frame, text='Add Record',
                   command=self.add).grid(row=7, column=2)

        self.message = Label(text='', fg='Red')
        self.message.grid(row=8, column=1)

        '''Database Table display box '''
        self.tree = ttk.Treeview(
            height=10, column=['', '', '', '', '', '', '', ''])
        self.tree.grid(row=9, column=0, columnspan=2)
        self.tree.heading('#0', text='ID')
        self.tree.column('#0', width=50)
        self.tree.heading('#1', text='ID Number')
        self.tree.column('#1', width=100)
        self.tree.heading('#2', text='Password')
        self.tree.column('#2', width=100)
        self.tree.heading('#3', text='Birth')
        self.tree.column('#3', width=90)
        self.tree.heading('#4', text='Event')
        self.tree.column('#4', width=150)
        self.tree.heading('#5', text='Medic')
        self.tree.column('#5', width=50, stretch=False)

        self.tree.heading('#6', text='Age')
        self.tree.column('#6', width=40, stretch=False)
        self.tree.heading('#7', text='Gender')
        self.tree.column('#7', width=60, stretch=False)

        def tick():
            d = datetime.datetime.now()
            today = '{:%B %d,%Y}'.format(d)

            mytime = time.strftime('%I:%M:%S%p')
            self.lblInfo.config(text=(mytime + '\t' + today))
            self.lblInfo.after(200, tick)

        self.lblInfo = Label(font=('arial', 20, 'bold'), fg='Dark green')
        self.lblInfo.grid(row=10, column=0, columnspan=2)
        tick()

        ''' Menu Bar '''
        Chooser = Menu()
        itemone = Menu()

        itemone.add_command(label='Add Record', command=self.add)
        itemone.add_command(label='Edit Record', command=self.edit)
        itemone.add_command(label='Delete Record', command=self.delet)
        itemone.add_separator()
        itemone.add_command(label='Refresh', command=self.refresh)
        itemone.add_command(label='Exit', command=self.ex)

        Chooser.add_cascade(label='File', menu=itemone)
        Chooser.add_command(label='Add', command=self.add)
        Chooser.add_command(label='Edit', command=self.edit)
        Chooser.add_command(label='Delete', command=self.delet)
        Chooser.add_command(label='Refresh', command=self.refresh)
        Chooser.add_command(label='Exit', command=self.ex)

        root.config(menu=Chooser)
        self.veiwing_records()

    ''' View Database Table'''

    def run_query(self, query, parameters=()):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query_result = cursor.execute(query, parameters)
            conn.commit()
        return query_result

    def veiwing_records(self):
        records = self.tree.get_children()
        for element in records:
            self.tree.delete(element)
        query = 'SELECT * FROM Userlist'
        db_table = self.run_query(query)
        for data in db_table:
            self.tree.insert('', 1000, text=data[0], values=data[1:])

    ''' Add New Record '''

    def validation(self):
        return len(self.IDnumber.get()) != 0 and len(self.Password.get()) != 0 and len(self.Birth.get()) != 0 and \
            len(self.Event.get()) != 0 and len(
                self.Medic.get()) != 0

    def add_record(self):
        if self.validation():
            query = 'INSERT INTO Userlist VALUES (NULL,?,?,?,?,?,?,?)'
            parameters = (self.IDnumber.get(), self.Password.get(), self.Birth.get(),
                          self.Event.get(), self.Medic.get(), None, None)
            self.run_query(query, parameters)
            self.message['text'] = 'Record {} added!'.format(
                self.IDnumber.get())

            self.IDnumber.delete(0, END)
            self.Password.delete(0, END)
            self.Birth.delete(0, END)
            self.Event.delete(0, END)
            self.Medic.delete(0, END)

        else:
            self.message['text'] = 'Fields not completed! Complete all fields...'

        self.veiwing_records()

    '''Function for using buttons'''

    def add(self):
        ad = tkinter.messagebox.askquestion(
            'Add Record', 'Do you want to add a New Record?')
        if ad == 'yes':
            self.add_record()

    ''' Deleting a Record '''

    def delete_record(self):
        # To clear output
        self.message['text'] = ''

        try:
            # why 1? --Can be anything
            self.tree.item(self.tree.selection())['values'][1]

        except IndexError as e:
            self.message['text'] = 'Please select a record to delete!'
            return

        # Again clear output
        self.message['text'] = ''
        # ???why text
        number = self.tree.item(self.tree.selection())['text']
        query = 'DELETE FROM Userlist WHERE ID = ?'
        # Why comma
        self.run_query(query, (number,))
        self.message['text'] = 'Record {} deleted!'.format(number)

        # Printing new database

        self.veiwing_records()

    # Function to add functionality in buttons

    def delet(self):
        de = tkinter.messagebox.askquestion(
            'Delete Record', 'Are you sure you want to delete this Record?')
        if de == 'yes':
            self.delete_record()

    '''EDIT RECORD'''

    '''CREATING A POP UP WINDOW FOR EDIT'''

    def edit_box(self):
        self.message['text'] = ''
        try:
            self.tree.item(self.tree.selection())['values'][0]

        except IndexError as e:
            self.message['text'] = 'Please select a Record to Edit!'
            return

        idnum = self.tree.item(self.tree.selection())['values'][0]
        pw = self.tree.item(self.tree.selection())['values'][1]
        birth = self.tree.item(self.tree.selection())['values'][2]
        Event = self.tree.item(self.tree.selection())['values'][3]
        Medic = self.tree.item(self.tree.selection())['values'][4]

        self.edit_root = Toplevel()
        self.edit_root.title('Edit Record')
        self.edit_root.geometry('305x355+600+200')

        Label(self.edit_root, text='Old IDnumber').grid(
            row=0, column=1, sticky=W)
        Entry(self.edit_root, textvariable=StringVar(self.edit_root, value=idnum), state='readonly').grid(row=0,
                                                                                                          column=2)
        Label(self.edit_root, text='New IDnumber').grid(
            row=1, column=1, sticky=W)
        new_idnum = Entry(self.edit_root, textvariable=StringVar(
            self.edit_root, value=idnum))
        new_idnum.grid(row=1, column=2)

        Label(self.edit_root, text='Old Password').grid(
            row=2, column=1, sticky=W)
        Entry(self.edit_root, textvariable=StringVar(self.edit_root, value=pw), state='readonly').grid(row=2,
                                                                                                       column=2)
        Label(self.edit_root, text='New Password').grid(
            row=3, column=1, sticky=W)
        new_pw = Entry(self.edit_root, textvariable=StringVar(
            self.edit_root, value=pw))
        new_pw.grid(row=3, column=2)

        Label(self.edit_root, text='Old Birth').grid(
            row=4, column=1, sticky=W)
        Entry(self.edit_root, textvariable=StringVar(self.edit_root, value=birth), state='readonly').grid(row=4,
                                                                                                          column=2)
        Label(self.edit_root, text='New Birth').grid(
            row=5, column=1, sticky=W)
        new_birth = Entry(self.edit_root, textvariable=StringVar(
            self.edit_root, value=birth))
        new_birth.grid(row=5, column=2)

        Label(self.edit_root, text='Old Event').grid(row=6, column=1, sticky=W)
        Entry(self.edit_root, textvariable=StringVar(self.edit_root, value=Event), state='readonly').grid(row=6,
                                                                                                          column=2)
        Label(self.edit_root, text='New Event').grid(row=7, column=1, sticky=W)
        new_Event = Entry(self.edit_root, textvariable=StringVar(
            self.edit_root, value=Event))
        new_Event.grid(row=7, column=2)

        Label(self.edit_root, text='Old Medic').grid(
            row=8, column=1, sticky=W)
        Entry(self.edit_root, textvariable=StringVar(self.edit_root, value=Medic), state='readonly').grid(row=8,
                                                                                                          column=2)
        Label(self.edit_root, text='New Medic').grid(
            row=9, column=1, sticky=W)
        new_Medic = Entry(self.edit_root, textvariable=StringVar(
            self.edit_root, value=Medic))
        new_Medic.grid(row=9, column=2)

        Button(self.edit_root, text='Save Changes', command=lambda: self.edit_record(new_idnum.get(), idnum, new_pw.get(), pw,
                                                                                     new_birth.get(), birth, new_Event.get(), Event, new_Medic.get(), Medic)).grid(row=12, column=2, sticky=W)

        self.edit_root.mainloop()

    def edit_record(self, new_idnum, idnum, new_pw, pw, new_birth, birth, new_Event, Event, new_Medic, Medic):
        query = 'UPDATE Userlist SET IDnumber=?, Password=?, Birth=?, Event=?, Medic=? WHERE ' \
                'IDnumber=? AND Password=? AND Birth=? AND Event=? AND Medic=? '

        parameters = (new_idnum, new_pw, new_birth, new_Event,
                      new_Medic, idnum, pw, birth, Event, Medic)
        self.run_query(query, parameters)
        self.edit_root.destroy()
        self.message['text'] = '{} details are changed to {}'.format(
            idnum, new_idnum)
        self.veiwing_records()

    def edit(self):
        ed = tkinter.messagebox.askquestion(
            'Edit Record', 'Want to Edit this Record?')
        if ed == 'yes':
            self.edit_box()

    '''Refresh'''

    def refresh(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
        query = 'SELECT Birth,IDnumber FROM Userlist'
        cursor.execute(query)
        data = cursor.fetchall()
        for row in data:
            today = datetime.date.today()
            year = row[0][0:4]
            month = row[0][5:7]
            day = row[0][8:10]
            age = today.year - int(year) - \
                ((today.month, today.day) < (int(month), int(day)))
            if row[1][1] == '1':
                gen = "Male"
            else:
                gen = "Female"
            self.refresh_record(age, gen, row[1])
        tkinter.messagebox.showinfo('Log', 'Refrashed!')

    def refresh_record(self, age, gen, idnum):
        query = 'UPDATE Userlist SET Age=?,Gender=? WHERE IDnumber=?'
        parameters = (age, gen, idnum)
        self.run_query(query, parameters)
        self.veiwing_records()

    def ex(self):
        exit = tkinter.messagebox.askquestion(
            'Exit Application', 'Are you sure you want to close this application?')
        if exit == 'yes':
            self.root.destroy()


'''MAIN'''

if __name__ == '__main__':
    root = Tk()
    # root.geometry('585x515+500+200')
    application = Hospital_Portal(root)
    root.mainloop()
